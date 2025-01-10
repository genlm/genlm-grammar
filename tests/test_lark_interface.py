import lark
import numpy as np
import string
from arsenal import colors

from genlm_cfg import BoolCFGLM, locally_normalize, EarleyLM, Earley
from genlm_cfg.lark_interface import LarkStuff


grammar1 = r"""
start: WS? "SELECT" WS select_expr WS "FROM" WS from_expr [WS "WHERE" WS bool_condition] [WS "GROUP BY" WS var_list] [WS "ORDER BY" WS orderby_expr] WS EOS
EOS: "</s>"
select_expr: STAR | select_list
bool_condition: bool_expr | "(" bool_condition WS "AND" WS bool_condition ")" | "(" bool_condition WS "OR" WS bool_condition ")"
bool_expr: var "=" value | var ">" value | var "<" value
from_expr: "data"
orderby_expr: var_list WS "ASC" | var_list WS "DESC"
select_list: select_var ("," WS select_var)*
var_list: var ("," WS var)*
select_var: var | "AVG(" var ")" | "MEDIAN(" var ")" | "COUNT(" var ")"
var: "age" | "gender" | "year" | "state_color" | "zipcode" | "vote" | "race_ethnicity"
value: NUMBER | "red" | "blue" | "white" | "black" | "latino" | "republican" | "democrat" | "male" | "female"
STAR: "*"
NUMBER: /\d+/
WS: /[ \t\f\r\n]/
"""


def test_parsing_basics():
    lark_stuff = LarkStuff(grammar1, cnf=True)

    text = 'SELECT state_color FROM data </s>'

    instance = lark.Lark(grammar1, parser='earley')

    tokens = list(instance.lex(text))

    g = lark_stuff.convert().renumber()
    assert g.in_cnf()  # lark returns a grammar in CNF

    tokens = ['WS', 'SELECT', 'WS', 'ZIPCODE', 'WS', 'FROM', 'WS', 'DATA', 'WS', 'EOS']

    assert g(tokens) > 0

    # print(g.cnf.prefix_grammar.trim().cnf)
    tokens = ['WS', 'SELECT', 'WS', 'ZIPCODE', 'WS', 'FROM', 'WS', 'DATA']

    assert g.prefix_weight(tokens) > 0

    ####
    # Now, we repeat the same as above without CNF conversion
    #
    # NOTE: the grammars are unfortunately not equivalent because of how we
    # assigned them weights.

    lark_stuff = LarkStuff(grammar1, cnf=False)

    text = 'SELECT state_color FROM data </s>'
    tokens = list(instance.lex(text))

    g = lark_stuff.convert().renumber()

    tokens = ['WS', 'SELECT', 'WS', 'ZIPCODE', 'WS', 'FROM', 'WS', 'DATA', 'WS', 'EOS']

    assert Earley(g)(tokens) > 0

    # print(g.cnf.prefix_grammar.trim().cnf)
    tokens = ['WS', 'SELECT', 'WS', 'ZIPCODE', 'WS', 'FROM', 'WS', 'DATA']

    assert Earley(g.prefix_grammar)(tokens) > 0


def test_char_level_cfg():
    lark_stuff = LarkStuff(grammar1)

    # this grammar is kind of silly - it requires a space at the front of the string
    text = 'SELECT state_color FROM data </s>'

    # tokens = list(lark_stuff.lex(text))
    # print(lark_stuff.parser.parse(tokens, 'start'))

    cfg = lark_stuff.char_cfg()

    # print(len(cfg.trim()))
    # print(len(cfg.cnf))

    assert cfg(text) > 0

    lm = EarleyLM(locally_normalize(cfg, tol=1e-40, maxiter=np.inf))

    p = lm.p_next('SELECT state_color FROM ').normalize()
    print(p)
    p.assert_equal({'d': 1})

    p = lm.p_next('S').normalize()
    print(p)
    p.assert_equal({'E': 1})

    p = lm.p_next('SELECT ').normalize()
    print(p)
    assert p.argmax() == '*'


def test_char_lm_basics1():
    lark_stuff = LarkStuff(
        r"""

        start: "SELECT" WS NAME WS "FROM" WS NAME WS EOS

        EOS: "</s>"
        NAME: /[A-Za-z][A-Za-z]?[A-Za-z]?[A-Za-z]?[A-Za-z]?/
        STAR: "*"
        WS: /[ ]/

        """
    )

    cfg_t = lark_stuff.char_cfg()

    pg = locally_normalize(cfg_t.cnf.trim()).prefix_grammar.trim()
    pg = pg.cnf

    assert pg('S') > 0
    assert pg('SEL') > 0


def test_char_lm_basics2():
    lark_stuff = LarkStuff(
        r"""
        start: NAME
        NAME: /(a|b)+/
        """
    )
    cfg_t = lark_stuff.char_cfg()
    pg = cfg_t.cnf.prefix_grammar.cnf.trim()
    pg.materialize(3).assert_equal(
        {
            (): 1,
            ('a',): 0.5,
            ('b',): 0.5,
            ('a', 'a'): 0.16666,
            ('a', 'b'): 0.16666,
            ('b', 'a'): 0.16666,
            ('b', 'b'): 0.16666,
            ('a', 'a', 'a'): 0.055555,
            ('a', 'a', 'b'): 0.055555,
            ('a', 'b', 'a'): 0.055555,
            ('a', 'b', 'b'): 0.055555,
            ('b', 'a', 'a'): 0.055555,
            ('b', 'a', 'b'): 0.055555,
            ('b', 'b', 'a'): 0.055555,
            ('b', 'b', 'b'): 0.055555,
        },
        tol=1e-4,
    )


def test_char_lm_basics3():
    lark_stuff = LarkStuff(
        r"""
        start: "SELECT" " " NAME " " "FROM"
        NAME: /b+/
        """
    )

    cfg_t = lark_stuff.char_cfg()

    cfg_t_lm = EarleyLM(locally_normalize(cfg_t, tol=1e-50))

    v = cfg_t_lm.p_next('SELECT bb').normalize()
    print(v)
    assert set(v.trim().keys()) == {' ', 'b'}

    char_cfg = lark_stuff.char_cfg()
    char_lm = EarleyLM(locally_normalize(char_cfg, tol=1e-50))

    v = char_lm.p_next('SELECT bb').normalize()
    print(v)
    assert set(v.trim().keys()) == {' ', 'b'}


def test_case_insensitive_char_proposal():
    grammar = r"""
    start: WS? "SELECT"i WS
    WS: /[ ]/
    """

    guide = EarleyLM(locally_normalize(LarkStuff(grammar).char_cfg()))

    assert guide.p_next('').trim().keys() == {'S', 's', ' '}
    assert guide.p_next('S').trim().keys() == {'E', 'e'}
    assert guide.p_next('s').trim().keys() == {'E', 'e'}


# def test_case_insensitive_expansion():
#    assert expand_case_insensitive('AND') == 'AND'
#    assert expand_case_insensitive('(?i:AND)') == '[aA][nN][dD]'
#
#    assert expand_case_insensitive('[aA][nN][dD]') == '[aA][nN][dD]'
#    assert expand_case_insensitive('(?i:[aA][nN][dD])') == '[aA][nN][dD]'
#
#    assert expand_case_insensitive('(?i:AND|OR)') == '[aA][nN][dD]|[oO][rR]'
#    assert expand_case_insensitive('(?i:[aA][nN][dD]|OR)') == '[aA][nN][dD]|[oO][rR]'
#    assert expand_case_insensitive('(?i:AND)|(?i:OR)') == '[aA][nN][dD]|[oO][rR]'
#
#    assert expand_case_insensitive('(?i:[aA][nN][d)') == '[aA][nN][[dD]'
#    assert expand_case_insensitive('(?i:[aA][nN][dD)') == '[aA][nN][[dD][dD]'
#    assert expand_case_insensitive('(?i:[aA][nN][dE])') == '[aA][nN][[dD][eE]]'
#    assert expand_case_insensitive('(?i:[aA][nN][dDE])') == '[aA][nN][[dD][dD][eE]]'
#
#    assert expand_case_insensitive('(?i:(?i:AND))') == '[aA][nN][dD]'
#    assert expand_case_insensitive('(?i:(?i:(?i:AND)))') == '[aA][nN][dD]'
#    assert (
#        expand_case_insensitive('(?i:(?i:(?i:AND)))(?i:(?i:AND))(?i:AND)')
#        == '[aA][nN][dD][aA][nN][dD][aA][nN][dD]'
#    )
#    assert (
#        expand_case_insensitive('(?i:(?i:(?i:AND)|(?i:OR)))') == '[aA][nN][dD]|[oO][rR]'
#    )
#
#    assert expand_case_insensitive('(?i:(AND|OR))') == '([aA][nN][dD]|[oO][rR])'
#    assert expand_case_insensitive('(?i:AND|(?i:OR))') == '[aA][nN][dD]|[oO][rR]'
#
#    assert (
#        expand_case_insensitive('(?i:[a-z][A-Z][a-zA-z])') == '[a-zA-Z][a-zA-Z][a-zA-Z]'
#    )
#    assert (
#        expand_case_insensitive('[a-z](?i:a[a-z]z)[a-z]') == '[a-z][aA][a-zA-Z][zZ][a-z]'
#    )
#
#    assert expand_case_insensitive('(?i:\n)') == '\n'
#    assert expand_case_insensitive('(?i:\\\\n)') == '\\\\[nN]'
#
#    sql_example_input = '(?:(?:(?:(?i:RIGHT)|(?i:FULL)|(?i:LEFT))(?:(?:[ \t\x0c\r\n])+(?i:OUTER))?|(?i:INNER)|(?:(?i:RIGHT)|(?i:FULL)|(?i:LEFT))|(?i:(?:(?i:OUTER))?))(?:[ \t\x0c\r\n])+)?(?i:JOIN)[ ]?'
#    sql_example_output = '(?:(?:(?:[rR][iI][gG][hH][tT]|[fF][uU][lL][lL]|[lL][eE][fF][tT])(?:(?:[ \t\x0c\r\n])+[oO][uU][tT][eE][rR])?|[iI][nN][nN][eE][rR]|(?:[rR][iI][gG][hH][tT]|[fF][uU][lL][lL]|[lL][eE][fF][tT])|(?:[oO][uU][tT][eE][rR])?)(?:[ \t\x0c\r\n])+)?[jJ][oO][iI][nN][ ]?'
#    assert expand_case_insensitive(sql_example_input) == sql_example_output


def test_github_issue_26_():
    # [2024-07-02 Tue] The original lark -> genlm_cfg.CFG translation of this
    # grammar had a nonterminal--terminal naming conflict.
    grammar = """
    start: x and x
    x: "b" | "a"
    and: " AND "
    """

    L = LarkStuff(grammar)

    cfg = L.char_cfg()

    assert cfg.V == {'A', 'N', 'D', 'a', 'b', ' '}

    cfg.language(100).assert_equal(
        {
            ('b', ' ', 'A', 'N', 'D', ' ', 'b'): 0.25,
            ('b', ' ', 'A', 'N', 'D', ' ', 'a'): 0.25,
            ('a', ' ', 'A', 'N', 'D', ' ', 'b'): 0.25,
            ('a', ' ', 'A', 'N', 'D', ' ', 'a'): 0.25,
        }
    )

    # The original failing example is below:

    grammar = """
    start: sent
    sent: "exists " var " . " sent
    | "forall " var " . " sent
    | "( " sent " )"
    | sent " AND " sent
    | sent " OR " sent
    | expr "(" var ")"
    | expr "(" var ", " var ")"
    | expr "(" var ", " const ")"
    var: "x" | "y" | "z" | "a" | "e" | "i"
    expr: "boy" | "girl"
    const: "Bill" | "Mary"
    """

    guide = BoolCFGLM(LarkStuff(grammar).char_cfg())

    guide.p_next('exists x . boy(x)').assert_equal({'▪': 1, ' ': 1})

    # The bug originally allowed 'a'
    guide.p_next('exists x . boy(x) ').assert_equal({'A': 1, 'O': 1})

    guide.p_next('exists x . boy(x) a').assert_equal({})


def test_lark_ignore():
    grammar = r"""
    start: "SELECT" NAME "FROM" NAME EOS
    NAME: /[A-Za-z][A-Za-z]?[A-Za-z]?[A-Za-z]?[A-Za-z]?/
    EOS: "</s>"
    WS: /[ ]/
    %ignore WS
    """

    guide = BoolCFGLM(LarkStuff(grammar).char_cfg())

    assert guide.p_next('').keys() == {'S', ' '}
    assert guide.p_next(' ').keys() == {'S'}
    assert guide.p_next(' S').keys() == {'E'}
    assert ' ' in guide.p_next(' SELECT').keys()
    assert ' ' not in guide.p_next(' SELECT ').keys()


# def test_char_cfg_delimiter():
#    grammar = r"""
#    start: "SELECT" NAME "FROM" NAME EOS
#    NAME: /[A-Za-z][A-Za-z]?[A-Za-z]?[A-Za-z]?[A-Za-z]?/
#    EOS: "</s>"
#    """
#
#    import warnings
#
#    with warnings.catch_warnings():
#        warnings.simplefilter('ignore')
#        guide = BoolCFGLM(LarkStuff(grammar).char_cfg(delimiter='[ \n]'))
#
#    assert guide.p_next('').keys() == {'S'}
#    assert guide.p_next(' ') == {}
#    assert guide.p_next('\n') == {}
#    assert guide.p_next('SELECT').keys() == {' ', '\n'}
#    assert not any(x in guide.p_next('SELECT ').keys() for x in (' ', '\n'))
#    assert not any(x in guide.p_next('SELECT\n').keys() for x in (' ', '\n'))
#    assert guide.p_next('SELECT\n\n') == {}
#    assert guide.p_next('SELECT x ').keys() == {'F'}
#    assert guide.p_next('SELECT x FROM').keys() == {' ', '\n'}


def test_char_cfg_charset():
    grammar = r'start: /[^a]+/'
    guide = BoolCFGLM(LarkStuff(grammar).char_cfg(charset='core'))
    have = set(guide.p_next('').trim().keys())
    want = set(string.printable) - {'a'}
    assert have == want, f'\n\nhave=\n{have}\n\nwant=\n{want}'
    print(colors.mark(True), repr(grammar))

    grammar = r'start: /.+/'
    guide = BoolCFGLM(LarkStuff(grammar).char_cfg(charset='core'))
    have = set(guide.p_next('').trim().keys())
    want = set(string.printable) - {
        '\n'
    }  # interegular does not default to multiline regular expressions
    assert have == want, f'\n\nhave=\n{have}\n\nwant=\n{want}'
    print(colors.mark(True), repr(grammar))


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
