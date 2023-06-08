# 베이즈 네트워크 기본 코드들

from probability import *
import random
from functools import reduce

class BayesNet:
    """베이즈 네트워크 클래스. 불리언 변수로만 구성된 버전"""

    def __init__(self, node_specs=None):
        """node_specs: 노드(BayesNode) 리스트. 노드의 순서는 부모가 자식보다 앞쪽에 위치하도록 정렬되어야 함.
        이 리스트의 각 원소는 BayesNode 생성에 필요한 (X, parents, cpt) 튜플로 구성됨."""
        self.nodes = []
        self.variables = []
        node_specs = node_specs or []
        for node_spec in node_specs:
            self.add(node_spec)

    def add(self, node_spec):
        """네트워크에 노드 추가.
        추가하는 노드의 부모가 이미 네트워크에 존재해야 하며,
        추가하는 노드의 변수는 네트워크에 존재하지 않아야 함."""
        node = BayesNode(*node_spec)
        assert node.variable not in self.variables
        assert all((parent in self.variables) for parent in node.parents)
        self.nodes.append(node)
        self.variables.append(node.variable)
        for parent in node.parents:
            self.variable_node(parent).children.append(node)

    def variable_node(self, var):
        """var에 해당하는 변수의 노드를 리턴함."""
        for n in self.nodes:
            if n.variable == var:
                return n
        raise Exception("No such variable: {}".format(var))

    def variable_values(self, var):
        """var의 도메인을 리턴함."""
        return [True, False]

    def __repr__(self):
        return 'BayesNet({0!r})'.format(self.nodes)


class BayesNode:
    """베이즈 네트워크의 노드 클래스.
    불리언 변수에 대한 조건부 확률 분포 P(X | parents)."""

    def __init__(self, X, parents, cpt):
        """X: 변수명,
        parents: 변수명 시퀀스 또는 공백으로 구분된 문자열
        cpt: 조건부 확률표. 다음 중 하나의 형태:

        * 숫자. 무조건부 확률 P(X=true). 부모가 없는 변수인 경우.

        * {v: p, ...} 형식의 딕셔너리. 조건부 확률 분포 P(X=true | parent=v) = p.
          부모가 하나뿐인 경우.

        * {(v1, v2, ...): p, ...} 형식의 딕셔너리. 조건부 확률 분포 P(X=true | parent1=v1, parent2=v2, ...) = p.
          딕셔너리의 키는 부모 개수 만큼의 값을 가져야 함. 가장 일반적인 형태.
          내부적으로는 이런 형식으로 값이 저장됨.
          
        P(X=true)를 이용하여 P(X=false)는 계산할 수 있으므로, cpt에는 거짓일 확률은 명시하지 않음.
        """
        if isinstance(parents, str):
            parents = parents.split()

        # cpt는 항상 {(v1, v2, ...): p, ...} 형식의 딕셔너리로 저장
        if isinstance(cpt, (float, int)):  # 부모 없음. 0-튜플
            cpt = {(): cpt}
        elif isinstance(cpt, dict):
            # 부모 하나. 1-튜플
            if cpt and isinstance(list(cpt.keys())[0], bool):
                cpt = {(v,): p for v, p in cpt.items()}

        assert isinstance(cpt, dict)
        for vs, p in cpt.items():
            assert isinstance(vs, tuple) and len(vs) == len(parents)
            assert all(isinstance(v, bool) for v in vs)
            assert 0 <= p <= 1

        self.variable = X
        self.parents = parents
        self.cpt = cpt
        self.children = []

    def p(self, value, event):
        """조건부 확률 P(X=value | parents=parent_values) 리턴
        parent_values: event에 있는 부모의 값들. event는 각 부모에 하나의 값을 배정해야 함.
        >>> bn = BayesNode('X', 'Burglary', {T: 0.2, F: 0.625})
        >>> bn.p(False, {'Burglary': False, 'Earthquake': True})
        0.375"""
        assert isinstance(value, bool)
        ptrue = self.cpt[event_values(event, self.parents)]
        return ptrue if value else 1 - ptrue

    def sample(self, event):
        """event의 부모 변수 값이 주어졌을 때 노드 변수의 조건부 확률 분포로부터 샘플링.
        조건부 확률에 따라 True/False를 무작위로 리턴"""
        return probability(self.p(True, event))

    def __repr__(self):
        return repr((self.variable, ' '.join(self.parents)))


def probability(p):
    """확률 p에 따라 True를 리턴함"""
    return p > random.uniform(0.0, 1.0)


T, F = True, False
