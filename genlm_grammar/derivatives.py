from dataclasses import dataclass, field
from functools import wraps
from re import L
from typing import Iterable, Callable, Literal
from collections import defaultdict


ANY_BYTE = frozenset(range(256))

BookKeepingTasks = set[
    Literal[
        "matches_empty",
        "possible_starts",
        "could_have_matches",
        "derivatives",
    ]
]


def all_complete() -> BookKeepingTasks:
    return {
        "matches_empty",
        "possible_starts",
        "could_have_matches",
        "derivatives",
    }


CURRENT_BOOK_KEEPER = None


@dataclass(slots=True)
class BookKeeping:
    matches_empty: bool = False
    possible_starts: frozenset[int] = ANY_BYTE
    could_have_matches: bool = False
    derivatives: dict[str, "Grammar"] = field(default_factory=dict)
    complete: BookKeepingTasks = field(default_factory=set)


GRAMMAR_KWARGS = dict(
    slots=True,
    frozen=True,
    eq=False,
)


def book_keeping_property(name):
    def f(self):
        global CURRENT_BOOK_KEEPER
        if name not in self.book_keeping.complete:
            key = (name, self)
            if CURRENT_BOOK_KEEPER is None:
                try:
                    CURRENT_BOOK_KEEPER = BookKeeper()
                    CURRENT_BOOK_KEEPER.request(key)
                    CURRENT_BOOK_KEEPER.run()
                finally:
                    CURRENT_BOOK_KEEPER = None
            else:
                return CURRENT_BOOK_KEEPER.get_value(key)
            assert name in self.book_keeping.complete
        return getattr(self.book_keeping, name)

    f.__name__ = name
    return property(f)


class Grammar:
    __slots__ = ("book_keeping",)

    def __init__(self):
        self.book_keeping: BookKeeping = BookKeeping()

    matches_empty = book_keeping_property("matches_empty")
    could_have_matches = book_keeping_property("could_have_matches")
    possible_starts = book_keeping_property("possible_starts")
    derivatives = book_keeping_property("derivatives")

    def force(self) -> "Grammar":
        return self

    # We cache creation of all grammar objects, so equality and
    # hashing are just by reference.
    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return object.__hash__(self)

    def __add__(self, other):
        return cat(self, other)

    def __or__(self, other):
        return union(self, other)


class Lazy(Grammar):
    __slots__ = ("__thunk",)

    def __init__(self, thunk: Callable[[], Grammar]):
        super().__init__()
        self.__thunk: Grammar | Callable[[], Grammar] = thunk

    @property
    def value(self) -> Grammar:
        if not isinstance(self.__thunk, Grammar):
            thunked: Grammar = self.__thunk()
            assert isinstance(thunked, Grammar)
            seen: set[Grammar] = {self}
            to_assign: list[Lazy] = [self]

            while True:
                if thunked in seen:
                    thunked = null
                    break
                seen.add(thunked)

                match thunked:
                    case Union(children) if self in children:
                        thunked = union(*(children - seen))
                    case Cat(left, right) if left == self or right == self:
                        thunked = null
                    case Lazy():
                        assert thunked is not self
                        if thunked.has_been_forced:
                            assert isinstance(thunked.__thunk, Grammar)
                            thunked = thunked.__thunk
                        else:
                            to_assign.append(thunked)
                            assert not isinstance(thunked.__thunk, Grammar)
                            thunked = thunked.__thunk()
                    case _:
                        break
            for t in to_assign:
                t.__thunk = thunked

        assert isinstance(self.__thunk, Grammar)
        assert not isinstance(self.__thunk, Lazy)
        return self.__thunk

    def force(self) -> "Grammar":
        return self.value

    @property
    def has_been_forced(self):
        return isinstance(self.__thunk, Grammar)

    def __repr__(self):
        # TODO: It would be nice to show the value here, but the problem is that
        # it might involve recursive references. Maybe import some pretty printing
        # code from Hypothesis for both printing the value and displaying the function
        # before it's evaluated?
        if self.has_been_forced:
            return "Lazy(evaluated)"
        else:
            return "Lazy(unevaluated)"


def lazy(fn):
    return Lazy(fn)


INPUT_CACHE = {}
OUTPUT_CACHE = {}


def cached(fn):
    @wraps(fn)
    def inner(*args):
        rewritten = []
        for a in args:
            if isinstance(a, set):
                a = frozenset(a)
            elif isinstance(a, list):
                a = tuple(a)
            rewritten.append(a)
        args = tuple(rewritten)

        key = (fn.__name__, *args)
        try:
            return INPUT_CACHE[key]
        except KeyError:
            pass
        args = [
            v.value if isinstance(v, Lazy) and v.has_been_forced else v for v in args
        ]
        result = fn(*args)
        if isinstance(result, Lazy) and result.has_been_forced:
            result = result.value
        if result not in OUTPUT_CACHE:
            do_initial_book_keeping(result)
            OUTPUT_CACHE[result] = result
        return result

    return inner


class Null(Grammar):
    def __repr__(self):
        return "null"


class Any(Grammar):
    """Matches any sequence of length n"""

    __slots__ = ("length",)
    __match_args__ = ("length",)

    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def __repr__(self):
        return f"any({self.length})"


class Chars(Grammar):
    __slots__ = __match_args__ = ("chars",)

    def __init__(self, chars: frozenset[int]):
        super().__init__()
        assert isinstance(chars, frozenset)
        for c in chars:
            assert isinstance(c, int)
        self.chars = chars

    def __repr__(self):
        return f"chars({bytes(self.chars)})"


class Epsilon(Grammar):
    def __repr__(self):
        return "epsilon"


class Cat(Grammar):
    __slots__ = __match_args__ = ("left", "right")

    def __init__(self, left: Grammar, right: Grammar):
        super().__init__()
        assert isinstance(left, Grammar)
        assert isinstance(right, Grammar)
        self.left = left
        self.right = right

    def __repr__(self):
        parts = [self.left]
        rest = self.right
        while isinstance(rest, Cat):
            parts.append(rest.left)
            rest = rest.right
        parts.append(rest)
        return f"cat({', '.join(map(repr, parts))})"


class Union(Grammar):
    __slots__ = __match_args__ = ("children",)

    def __init__(self, children: frozenset[Grammar]):
        super().__init__()
        for c in children:
            assert isinstance(c, Grammar)
        self.children = children

    def __repr__(self):
        children = list(map(repr, self.children))
        children.sort(key=lambda s: (len(s), s))
        return f"union({', '.join(children)})"


def do_initial_book_keeping(grammar):
    match grammar:
        case Any(k):
            assert k > 0
            grammar.book_keeping.matches_empty = False
            grammar.book_keeping.could_have_matches = True
            grammar.book_keeping.possible_starts = ANY_BYTE
            grammar.book_keeping.derivatives = {c: any(k - 1) for c in ANY_BYTE}
            grammar.book_keeping.complete = all_complete()
        case Chars(cs):
            assert cs
            grammar.book_keeping.matches_empty = False
            grammar.book_keeping.could_have_matches = True
            grammar.book_keeping.possible_starts = cs
            grammar.book_keeping.derivatives = {c: epsilon for c in cs}
            grammar.book_keeping.complete = all_complete()


@cached
def any(n: int):
    if n == 0:
        return epsilon
    return Any(n)


@cached
def chars(cs):
    cs = frozenset(cs)
    match len(cs):
        case 0:
            return null
        case 256:
            return dot
        case _:
            return Chars(cs)


@cached
def char(c):
    return chars(frozenset((c,)))


@cached
def _cat(left, right):
    match (left, right):
        case (Epsilon(), right):
            return right
        case (left, Epsilon()):
            return left
        case (Null(), _) | (_, Null()):
            return null
        case (Cat(u, v), right):
            return _cat(u, _cat(v, right))
        case (Any(m), Any(n)):
            return Any(m + n)
        case _:
            return Cat(left, right)


@cached
def cat(*args):
    match args:
        case ():
            return epsilon
        case (x,):
            return x
        case _:
            result = args[-1]
            for v in reversed(args[:-1]):
                result = _cat(v, result)
            return result


@cached
def literal(s: bytes) -> Grammar:
    return cat(*[char(c) for c in s])


@cached
def seq(child):
    result = lazy(lambda: union(epsilon, cat(child, result)))
    return result


@cached
def optional(child):
    return union(child, epsilon)


@cached
def union(*args: Grammar) -> Grammar:
    single_characters = set()
    flattened = set()
    stack: list[Iterable[Grammar]] = [args]
    deltas = set()
    has_epsilon = False
    while stack:
        for child in stack.pop():
            match child:
                case Chars(cs):
                    single_characters.update(cs)
                case Union(children):
                    stack.append(children)
                case Null():
                    continue
                case Epsilon():
                    has_epsilon = True
                case Any(1):
                    # FIXME: This is a slightly stupid way to do it
                    single_characters.update(range(256))
                case _:
                    flattened.add(child)

    if has_epsilon:
        flattened.add(epsilon)
    else:
        flattened.update(deltas)
    if not (single_characters or flattened):
        return null
    if single_characters:
        flattened.add(chars(single_characters))
    match len(flattened):
        case 0:
            return null
        case 1:
            return list(flattened)[0]
        case _:
            return Union(frozenset(flattened))


null = Null()


null.book_keeping.possible_starts = frozenset()
null.book_keeping.complete = all_complete()


epsilon = Epsilon()

epsilon.book_keeping.possible_starts = frozenset()
epsilon.book_keeping.matches_empty = True
epsilon.book_keeping.could_have_matches = True
epsilon.book_keeping.complete = all_complete()

dot = any(1)


def compact(grammar):
    if not grammar.could_have_matches:
        return null
    if not grammar.possible_starts:
        assert grammar.matches_empty
        return epsilon
    return grammar


def derivative(grammar: Grammar, c: int) -> Grammar:
    if c not in grammar.possible_starts:
        return null
    return compact(grammar.derivatives.get(c, null))


class BookKeeper:
    def __init__(self):
        self.targets: set[tuple[str, Grammar]] = set()
        self.watches = defaultdict(set)
        self.dirty = set()
        self.values_requested = set()

    def run(self):
        while self.dirty:
            needs_recalculation = self.dirty
            self.dirty = set()

            for target in needs_recalculation:
                if self.is_complete(target):
                    continue
                self.values_requested.clear()
                property_name, grammar = target
                if isinstance(grammar, Lazy):
                    value = self.get_value((property_name, grammar.value))
                else:
                    value = getattr(self, "calc_" + property_name)(grammar)
                self.set_value(target, value)
                for v in self.values_requested:
                    self.dependency(_from=target, to=v)
        for target in self.targets:
            self.mark_complete(target)

    def calc_matches_empty(self, grammar):
        match grammar:
            case Cat(left, right):
                return self.matches_empty(left) and self.matches_empty(right)
            case Union(children):
                resolved = []
                unresolved = []

                for c in children:
                    if c.book_keeping.matches_empty:
                        return True
                    elif isinstance(c, Lazy) and not c.has_been_forced:
                        unresolved.append(c)
                    else:
                        resolved.append(c)
                for c in resolved:
                    if self.matches_empty(c):
                        return True
                for c in unresolved:
                    if self.matches_empty(c):
                        return True
                return False
            case _:
                # Everything else should be calculated in initial_book_keeping
                raise AssertionError(grammar)

    def calc_possible_starts(self, grammar):
        match grammar:
            case Cat(left, right):
                if self.matches_empty(left):
                    return self.possible_starts(left) | self.possible_starts(right)
                else:
                    return self.possible_starts(left)
            case Union(children):
                result = set()
                for c in children:
                    result |= self.possible_starts(c)
                    if len(result) == 256:
                        break
                return frozenset(result)
            case _:
                raise AssertionError(grammar)

    def calc_could_have_matches(self, grammar):
        if self.matches_empty(grammar):
            return True
        if not self.possible_starts(grammar):
            return False
        match grammar:
            case Cat(left, right):
                return self.could_have_matches(left) and self.could_have_matches(right)
            case Union(children):
                # TODO: First check if anything has been established to match
                # empty before adding dependencies.
                for c in children:
                    if self.could_have_matches(c):
                        return True
                return False
            case _:
                raise AssertionError(grammar)

    def calc_derivatives(self, grammar):
        assert isinstance(grammar, Grammar)

        match grammar:
            case Union(children):
                result = {}
                for c in self.possible_starts(grammar):
                    child = union(*[derivative(child, c).force() for child in children])
                    if child != null:
                        result[c] = child
                return result
            case Cat(left, right):
                result = {}
                for c in self.possible_starts(left):
                    result[c] = cat(derivative(left, c).force(), right)
                if self.matches_empty(left):
                    for c in self.possible_starts(right):
                        if c in result:
                            result[c] = union(result[c], derivative(right, c).force())
                        else:
                            result[c] = derivative(right, c)
                return result
            case Lazy():
                return grammar.value.derivatives
            case _:
                raise AssertionError(grammar)

    def matches_empty(self, grammar):
        return self.get_value(("matches_empty", grammar))

    def possible_starts(self, grammar):
        return self.get_value(("possible_starts", grammar))

    def could_have_matches(self, grammar):
        return self.get_value(("could_have_matches", grammar))

    def request(self, target):
        if self.is_complete(target):
            return
        if target not in self.targets:
            self.targets.add(target)
            self.dirty.add(target)

    def dependency(self, _from, to):
        assert not self.is_complete(_from)
        if self.is_complete(to):
            return
        self.request(to)
        self.watches[to].add(_from)

    def is_complete(self, target):
        property_name, grammar = target
        return property_name in grammar.book_keeping.complete

    def mark_complete(self, target):
        property_name, grammar = target
        return grammar.book_keeping.complete.add(property_name)

    def get_value(self, target):
        self.values_requested.add(target)
        property_name, grammar = target
        return getattr(grammar.book_keeping, property_name)

    def set_value(self, target, value):
        property_name, grammar = target
        prev = getattr(grammar.book_keeping, property_name)
        if prev != value:
            setattr(grammar.book_keeping, property_name, value)
            self.dirty.update(self.watches[target])


def matches(grammar, string):
    for c in string:
        grammar = derivative(grammar, c)
    return grammar.matches_empty
