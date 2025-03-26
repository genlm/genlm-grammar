from dataclasses import dataclass, field
from functools import wraps
from typing import Iterable, Callable, Literal
from collections import defaultdict


ANY_BYTE = frozenset(range(256))

BookKeepingTasks = set[
    Literal["matches_empty", "possible_starts", "could_have_matches"]
]


@dataclass(slots=True)
class BookKeeping:
    matches_empty: bool = False
    possible_starts: frozenset[int] = ANY_BYTE
    could_have_matches: bool = False
    complete: BookKeepingTasks = field(default_factory=set)


GRAMMAR_KWARGS = dict(
    slots=True,
    frozen=True,
    eq=False,
)


def book_keeping_property(name):
    def f(self):
        if name not in self.book_keeping.complete:
            bk = BookKeeper()
            bk.request((name, self))
            bk.run()
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

    # We cache creation of all grammar objects, so equality and
    # hashing are just by reference.
    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return object.__hash__(self)


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


null = Null()


def all_complete() -> BookKeepingTasks:
    return {
        "matches_empty",
        "possible_starts",
        "could_have_matches",
    }


null.book_keeping.possible_starts = frozenset()
null.book_keeping.complete = all_complete()


class Epsilon(Grammar):
    def __repr__(self):
        return "epsilon"


epsilon = Epsilon()

epsilon.book_keeping.possible_starts = frozenset()
epsilon.book_keeping.matches_empty = True
epsilon.book_keeping.could_have_matches = True
epsilon.book_keeping.complete = all_complete()


def do_initial_book_keeping(grammar):
    match grammar:
        case Any(k):
            assert k > 0
            grammar.book_keeping.matches_empty = False
            grammar.book_keeping.could_have_matches = True
            grammar.book_keeping.possible_starts = ANY_BYTE
            grammar.book_keeping.complete = all_complete()
        case Chars(cs):
            assert cs
            grammar.book_keeping.matches_empty = False
            grammar.book_keeping.could_have_matches = True
            grammar.book_keeping.possible_starts = cs
            grammar.book_keeping.complete = all_complete()
        case Delta():
            grammar.book_keeping.possible_starts = frozenset()


class Any(Grammar):
    """Matches any sequence of length n"""

    __slots__ = ("length",)
    __match_args__ = ("length",)

    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def __repr__(self):
        return f"any({self.length})"


@cached
def any(n: int):
    if n == 0:
        return epsilon
    return Any(n)


dot = any(1)


class Chars(Grammar):
    __slots__ = __match_args__ = ("chars",)

    def __init__(self, chars: frozenset[int]):
        super().__init__()
        self.chars = chars

    def __repr__(self):
        return f"chars({bytes(self.chars)})"


@cached
def chars(cs):
    match len(cs):
        case 0:
            return null
        case 256:
            return dot
        case _:
            return Chars(cs)


class Cat(Grammar):
    __slots__ = __match_args__ = ("left", "right")

    def __init__(self, left: Grammar, right: Grammar):
        super().__init__()
        self.left = left
        self.right = right

    def __repr__(self):
        parts = [self.left]
        rest = self.right
        while isinstance(rest, Cat):
            parts.append(rest.left)
            rest = rest.right
        parts.append(rest)
        return f'cat({", ".join(map(repr, parts))})'


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


class Union(Grammar):
    __slots__ = __match_args__ = ("children",)

    def __init__(self, children: frozenset[Grammar]):
        super().__init__()
        self.children = children

    def __repr__(self):
        children = list(map(repr, self.children))
        children.sort(key=lambda s: (len(s), s))
        return f'union({", ".join(children)})'


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
                case Delta():
                    deltas.add(child)
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


class Delta(Grammar):
    """Matches the empty string if child does, otherwise matches nothing."""

    __slots__ = __match_args__ = ("child",)

    def __init__(self, child: Grammar):
        super().__init__()
        self.child = child

    def __repr__(self):
        return f"delta({self.child})"


@cached
def delta(grammar: Grammar) -> Grammar:
    match grammar:
        case Epsilon():
            return epsilon
        case Null() | Chars() | Any():
            return null
        case Union(children):
            return union(*map(delta, children))
        case _:
            return Delta(grammar)


DERIVATIVE_CACHE = {}


@cached
def derivative(grammar: Grammar, c: int) -> Grammar:
    match grammar:
        case Epsilon() | Null():
            return null
        case Chars(cs):
            if c in cs:
                return epsilon
            else:
                return null
        case Any(n):
            return any(n - 1)
        case Union(children):
            return union(*[derivative(child, c) for child in children])
        case Cat(left, right):
            return union(
                cat(delta(left), lazy(lambda: derivative(right, c))),
                cat(derivative(left, c), right),
            )
        case Lazy():
            if grammar.has_been_forced:
                return derivative(grammar.value, c)
            else:
                return lazy(lambda: derivative(grammar.value, c))
        case _:
            raise AssertionError(grammar)


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
            case Delta(child):
                return self.matches_empty(child)
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
            case Delta(child):
                # Because Delta matches something iff it matches empty and we checked
                # that above.
                return False
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
