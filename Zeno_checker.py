import xml.etree.ElementTree as ET
from lark import Lark, Tree, Token
import re
from itertools import combinations, permutations, product
from itertools import chain
import sys

####################################################
## Function for parsing XML and logic expressions ##
####################################################

grammar = r"""
    ?start: expr
    ?expr: expr "&&" expr   -> and_expr
         | expr "||" expr   -> or_expr
         | "(" expr ")"
         | /[^()&|]+/
    %ignore /\s+/
"""

parser = Lark(grammar, start="expr")


_op_re = re.compile(r'^\s*'
                    # lhs: identifier or integer (allow negative ints)
                    r'([A-Za-z_][A-Za-z0-9_]*|-?\d+)'
                    r'\s*(>=|<=|==|!=|=|>|<|\+|-|\*|/)\s*'          # op
                    # rhs: identifier or integer
                    r'([A-Za-z_][A-Za-z0-9_]*|-?\d+)'
                    r'\s*$')


def split_op(expr: str):
    m = _op_re.match(expr)
    if not m:
        raise ValueError(f"Cannot parse atomic expression: {expr!r}")
    lhs, op, rhs = m.group(1), m.group(2), m.group(3)
    return lhs, op, rhs


def eval_dfs(node, var, var_val, const: dict, depth=0):
    if isinstance(node, Tree):
        g, d = node.children
        bg = eval_dfs(g, var, var_val, const, depth + 1)
        bd = eval_dfs(d, var, var_val, const, depth + 1)
        if node.data == "and_expr":
            return bg and bd
        elif node.data == "or_expr":
            return bg or bd
    else:  # it's a Token (leaf)
        expr = str(node).strip()
        lhs, op, rhs = split_op(expr)
        if not (lhs == var) and not (rhs == var):
            return True
        if lhs.isdigit():
            lhs_val = int(lhs)
            if rhs == var:
                rhs_val = var_val
                # print(lhs + op + rhs)
                if op == '>':
                    return lhs_val > rhs_val
                elif op == '<':
                    return lhs_val < rhs_val
                elif op == '>=':
                    return lhs_val >= rhs_val
                elif op == '<=':
                    return lhs_val <= rhs_val
                elif op == '==':
                    return lhs_val == rhs_val
                elif op == '!=':
                    return lhs_val != rhs_val
                else:
                    raise ValueError(f"Unknown operator: {op}")
            else:
                return True
        else:
            try:
                lhs_val = calculate_value(lhs, const)
                if rhs == var:
                    rhs_val = var_val
                    # print(lhs + op + rhs)
                    if op == '>':
                        return lhs_val > rhs_val
                    elif op == '<':
                        return lhs_val < rhs_val
                    elif op == '>=':
                        return lhs_val >= rhs_val
                    elif op == '<=':
                        return lhs_val <= rhs_val
                    elif op == '==':
                        return lhs_val == rhs_val
                    elif op == '!=':
                        return lhs_val != rhs_val
                    else:
                        raise ValueError(f"Unknown operator: {op}")
                else:
                    return True
            except ValueError:
                # print(const)
                if (rhs.isdigit() or rhs in const.keys()) and lhs == var:
                    # print(lhs + '/' + op + '/' + rhs)
                    if op == '>':
                        return eval_dfs(rhs + ' ' + '<' + ' ' + lhs, var, var_val, const, depth)
                    elif op == '<':
                        return eval_dfs(rhs + ' ' + '>' + ' ' + lhs, var, var_val, const, depth)
                    elif op == '>=':
                        return eval_dfs(rhs + ' ' + '<=' + ' ' + lhs, var, var_val, const, depth)
                    elif op == '<=':
                        return eval_dfs(rhs + ' ' + '>=' + ' ' + lhs, var, var_val, const, depth)
                    else:
                        return eval_dfs(rhs + ' ' + op + ' ' + lhs, var, var_val, const, depth)
                else:
                    raise ValueError(f"Cannot evaluate expression: {expr!r}")


def id_to_number(s):
    return int(s.replace('id', ''))


def calculate_value(s: str, const_dict: dict):
    s = s.strip()
    try:
        g, op, d = split_op(s)
        val_g = calculate_value(g, const_dict)
        val_d = calculate_value(d, const_dict)
        if op == '+':
            return val_g + val_d
        elif op == '-':
            return val_g - val_d
        elif op == '*':
            return val_g * val_d
        elif op == '/':
            return val_g / val_d
        else:
            raise ValueError(f"Unknown operator: {op}")
    except ValueError:
        if s.isdigit():
            return int(s)
        elif s in const_dict.keys():
            return const_dict[s]
        else:
            raise ValueError(f"Cannot calculate: {s!r}")

###########################
### GET GRAPHs from XML ###
###########################


def get_info(xmltree):
    text = (xmltree.findall('./declaration')[0]).text
    lines = text.splitlines()
    const = {}
    nb_dupl = 0
    Sync_dict = {}
    for line in lines:
        if len(line) == 0:
            continue
        if len(line) > 1 and line[0] == '/' and line[1] == '/':
            continue
        line = line.partition(';')[0]
        words = line.split()

        if words[0] == 'const' and words[1] == 'int':
            pre_text = line.strip(' ;')
            var, _, val = split_op(pre_text.removeprefix('const int '))
            const[var] = int(val)

        if words[0] == 'typedef':
            a, b = [g.strip() for g in re.search(
                r'int\[\s*([^,]+)\s*,\s*([^\]]+)\]', line).groups()]
            nb_dupl = calculate_value(b, const) - calculate_value(a, const) + 1

        if words[0] == 'chan':
            pre_list = line.strip(' ;').removeprefix('chan').split(',')
            for chan in pre_list:
                if '[' in chan and ']' in chan:
                    chan_id = chan.strip().split('[')[0]
                    nb_chan = calculate_value(
                        chan.strip().split('[')[1].strip(']'), const)
                    Sync_dict[chan_id] = []
                else:
                    chan_id = chan.strip()
                    Sync_dict[chan_id] = []

        if words[0] == 'urgent' and words[1] == 'chan':
            pre_list = line.strip(' ;').removeprefix(
                'urgent chan').split(',')
            for chan in pre_list:
                if '[' in chan and ']' in chan:
                    chan_id = chan.strip().split('[')[0]
                    nb_chan = calculate_value(
                        chan.strip().split('[')[1].strip(']'), const)
                    Sync_dict[chan_id] = []
                else:
                    chan_id = chan.strip()
                    Sync_dict[chan_id] = []
    return nb_dupl, Sync_dict, const


def get_graph(xmltree):
    templates = xmltree.findall('.//template')
    duplicate_nb, Sync_dict, const = get_info(xmltree)
    Graph_dict = {}
    Trans_dict = {}
    Loc_dict = {}
    clock_dict = {}
    for i, t in enumerate(templates):
        count = len(t.findall('location'))
        Graph_dict[i] = [[0 for _ in range(count)] for _ in range(count)]
        clock_dict[i] = []

        declaration = t.find('declaration')
        if declaration is not None:
            decl_text = declaration.text
            lines = decl_text.splitlines()
            for line in lines:
                if len(line) == 0:
                    continue
                line = line.partition(';')[0]
                words = line.split()
                if words[0] == 'const' and words[1] == 'int':
                    pre_text = line.strip(' ;')
                    var, _, val = split_op(pre_text.removeprefix('const int '))
                    const[var] = int(val)
                if words[0] == 'clock':
                    pre_list = line.strip(' ;').removeprefix(
                        'clock').split(',')
                    for clk in pre_list:
                        clk_id = clk.strip()
                        clock_dict[i].append(clk_id)

        cpt = 0
        for loc in t.findall('location'):
            loc_id = id_to_number(loc.get('id'))
            Loc_dict[loc_id] = (cpt, [])
            for label in loc.findall('label'):
                if label.get('kind') == 'invariant':
                    inv = parser.parse(label.text)
                    Loc_dict[loc_id][1].append(inv)
            cpt += 1

        for tr in t.findall('transition'):
            src = id_to_number(tr.find('source').get('ref'))
            dst = id_to_number(tr.find('target').get('ref'))
            id_nb = str(src) + ',' + str(dst)
            Graph_dict[i][Loc_dict[src][0]][Loc_dict[dst][0]] = id_nb
            Trans_dict[id_nb] = [None, []]  # [guard, [assignments]]
            for label in tr.findall('label'):

                if label.get('kind') == 'synchronisation':
                    sync_id = (label.text).split('[')[0].strip('!?')
                    if sync_id not in Sync_dict:
                        Sync_dict[sync_id] = []
                    Sync_dict[sync_id].append(id_nb)

                if label.get('kind') == 'guard':
                    try:
                        guard = parser.parse(label.text)
                        Trans_dict[id_nb][0] = guard
                    except:
                        continue

                if label.get('kind') == 'assignment':
                    assignments = (label.text).strip('\n').split(',')
                    for assignment in assignments:
                        try:
                            var, _, val = split_op(assignment)
                            if int(val) == 0:
                                Trans_dict[id_nb][1].append(var)
                        except ValueError:
                            continue

    return Graph_dict, Trans_dict, Sync_dict, Loc_dict, const, clock_dict


##########################
### Non-Zeno Algorithm ###
##########################


def find_cycles(adj_matrix,):
    n = len(adj_matrix)
    cycles = set()  # use set to avoid duplicates

    def dfs(start, current, path, visited):
        for neighbor in range(n):
            if adj_matrix[current][neighbor] != 0:
                if neighbor == start and len(path) > 0:
                    # Found a cycle
                    cycle = tuple(
                        sorted(path + [adj_matrix[current][neighbor]]))
                    cycles.add(cycle)
                elif neighbor not in visited:
                    dfs(start, neighbor, path +
                        [adj_matrix[current][neighbor]], visited | {neighbor})

    for v in range(n):
        dfs(v, v, [], {v})

    return [list(c) for c in cycles]


def loop_is_non_zeno(loop, trans_dict, clock_list, loc_dict, const):
    for clk in clock_list:
        v = False
        for trans in loop:
            # check if clk is reset
            if clk in trans_dict[trans][1]:
                v = True
                break
        if v == False:
            continue
        v = False
        for trans in loop:
            guard = trans_dict[trans][0]
            if guard is not None:
                if eval_dfs(guard, clk, 0, const) == False:
                    v = True
                    break
            src, dst = trans.split(',')
            inv1_tab = loc_dict[int(src)][1]
            if inv1_tab != []:
                inv = inv1_tab[0]
                if eval_dfs(inv, clk, 0, const) == False:
                    v = True
                    break
            inv2_tab = loc_dict[int(dst)][1]
            if inv2_tab != []:
                inv = inv2_tab[0]
                if eval_dfs(inv, clk, 0, const) == False:
                    v = True
                    break
        if v == True:
            return True
    return False


def get_template_nb(cycle, Graph_dict):
    trans = cycle[0]
    for i, t_matrix in Graph_dict.items():
        if trans in list(chain.from_iterable(t_matrix)):  # flatten
            return i
    raise ValueError("Template number not found for the given cycle.")


def get_couples(cycle_set, Graph_dict):
    couples = list(combinations(cycle_set, 2))
    f_couples = []
    for couple in couples:
        c1, c2 = couple
        template_nb_1 = get_template_nb(c1, Graph_dict)
        template_nb_2 = get_template_nb(c2, Graph_dict)
        if template_nb_1 != template_nb_2:
            f_couples.append(set(couple))
    return f_couples


def get_cycle_sets(cycle_dict, sync_dict, Graph_dict):
    couple_cycle_sets = []
    for sync in sync_dict.values():
        cycle_set = set()
        for trans in sync:
            for cycle in cycle_dict.keys():
                if trans in cycle:
                    cycle_set.add(cycle)
        couples_from_set = get_couples(cycle_set, Graph_dict)
        for couples in couples_from_set:
            if couples not in couple_cycle_sets:
                couple_cycle_sets.append(couples)

    # Getting single loops
    loops = list(cycle_dict.keys())
    for sync in sync_dict.values():
        for trans in sync:
            for cycle in cycle_dict.keys():
                if trans in cycle and cycle in loops:
                    loops.remove(cycle)
    return couple_cycle_sets, loops


def find_cycles_system(Graph_dict):
    cycles = []
    for matrix in Graph_dict.values():
        cycle = find_cycles(matrix)
        cycles = cycles + cycle
    return cycles


def filter_unsafe_loops_couples(cycles_sets, cycle_dict):
    unsafe_cycle_sets = []
    for cycle_set in cycles_sets:
        cycle_a, cycle_b = list(cycle_set)
        if cycle_dict[cycle_a] == False and cycle_dict[cycle_b] == False:
            unsafe_cycle_sets.append(cycle_set)
    return unsafe_cycle_sets


def filter_unsafe_loops(cycles, cycle_dict):
    unsafe_cycles = []
    for cycle in cycles:
        if cycle_dict[cycle] == False:
            unsafe_cycles.append(cycle)
    return unsafe_cycles


### Main function, following algorithm to find unsafe cycles ###

def get_unsafes(xmltree):
    Graph_dict, Trans_dict, Sync_dict, Loc_dict, const, clock_dict = get_graph(
        xmltree)
    cycles = find_cycles(Graph_dict[0])
    cycle_dict = {}
    for cycle in cycles:
        v = False
        for clock in clock_dict.values():
            if loop_is_non_zeno(cycle, Trans_dict, clock, Loc_dict, const) == True:
                v = True
                break
        cycle_dict[tuple(cycle)] = v
    print('non zeno loops:', [c for c, v in cycle_dict.items() if v is False])
    cycle_sets, cycles = get_cycle_sets(cycle_dict, Sync_dict, Graph_dict)
    unsafe_cycle_sets = filter_unsafe_loops_couples(cycle_sets, cycle_dict)
    unsafe_cycles = filter_unsafe_loops(cycles, cycle_dict)
    return unsafe_cycle_sets, unsafe_cycles

####

############
### MAIN ###
############


def __main__(argv):
    tree = ET.parse(argv[1])
    unsafe_cycle_sets, unsafe_cycles = get_unsafes(tree)
    if unsafe_cycle_sets == [] and unsafe_cycles == []:
        print("No unsafe cycles found. Sufficient non-Zeno")
    else:
        print("Unsafe cycle sets:")
        if unsafe_cycle_sets == []:
            print("None")
        else:
            for cycle_set in unsafe_cycle_sets:
                print(cycle_set)
        print("Unsafe cycles:")
        if unsafe_cycles == []:
            print("None")
        else:
            for cycle in unsafe_cycles:
                print(cycle)


if __name__ == "__main__":
    __main__(sys.argv)
