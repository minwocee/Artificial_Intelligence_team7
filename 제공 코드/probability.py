# 확률적 추론을 위한 기본 코드들

import numpy as np
from collections import defaultdict


class ProbDist:
    """이산 확률 분포를 정의하는 클래스.
    생성자에 확률변수(random variable)를 정의함.
    이후에 확률변수 각 값에 대한 확률값을 지정함(딕셔너리 이용)."""

    def __init__(self, var_name='?', freq=None):
        """var_name: 확률변수. freq가 None이 아니라면, (확률변수의 값, 빈도) 쌍으로 구성된 딕셔너리여야 함.
        이후, 빈도를 바탕으로 합이 1이 되는 확률이 되도록 정규화함."""
        self.prob = {} # 실제 확률값은 여기에 저장됨
        self.var_name = var_name
        self.values = []
        if freq:
            for (v, p) in freq.items():
                self[v] = p
            self.normalize()

    def __getitem__(self, val):
        """확률변수의 값(val)이 주어지면, 확률 P(val) 리턴."""
        try:
            return self.prob[val]
        except KeyError:
            return 0

    def __setitem__(self, val, p):
        """확률변수의 값 val에 대한 확률 P(val) = p로 설정"""
        if val not in self.values:
            self.values.append(val)
        self.prob[val] = p

    def normalize(self):
        """모든 확률값의 합이 1이 되도록 정규화하여 정규화된 분포를 리턴함.
        확률값 합이 0이면, ZeroDivisionError가 발생됨."""
        total = sum(self.prob.values())
        if not np.isclose(total, 1.0):
            for val in self.prob:
                self.prob[val] /= total
        return self

    def show_approx(self, numfmt='{:.3g}'):
        """확률변수의 값에 따라 정렬하고 확률값은 반올림하여 리턴함."""
        return ', '.join([('{}: ' + numfmt).format(v, p) for (v, p) in sorted(self.prob.items())])

    def __repr__(self):
        return "P({})".format(self.var_name)


class JointProbDist(ProbDist):
    """변수 집합에 대한 이산 확률 분포(결합 확률 분포) 클래스. ProbDist의 서브 클래스."""

    def __init__(self, variables):
        """variables: 대상 확률변수들."""
        self.prob = {}
        self.variables = variables
        self.vals = defaultdict(list) # k: [v1, v2, ...]

    def __getitem__(self, values):
        """주어진 변수 값들에 대한 결합 확률 리턴."""
        values = event_values(values, self.variables)
        return ProbDist.__getitem__(self, values)

    def __setitem__(self, values, p):
        """확률값 P(values) = p로 설정. values는 각 변수에 대해 하나의 값을 가진 튜플/딕셔너리.
        또한 각 변수에 대해 지금까지 관측된 각 변수의 값을 기록함."""
        values = event_values(values, self.variables)
        self.prob[values] = p
        for var, val in zip(self.variables, values):
            if val not in self.vals[var]:
                self.vals[var].append(val)

    def values(self, var):
        """변수에 대해 가능한 값의 집합을 리턴."""
        return self.vals[var]

    def __repr__(self):
        return "P({})".format(self.variables)


def event_values(event, variables):
    """이벤트(event)에 존재하는 변수들(variables)에 대한 값들의 튜플을 리턴함."""
    if isinstance(event, tuple) and len(event) == len(variables):
        return event
    else:
        return tuple([event[var] for var in variables])


def extend(s, var, val):
    """딕셔너리 s를 복사하고 var에 값 val을 세팅하여 확장하여 그 복사본을 리턴함."""
    return {**s, var: val}
