import numpy as np

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
from collections import defaultdict

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

def enumerate_joint(variables, e, P):
    """증거 e에 포함되지 않은 확률분포 P의 나머지 변수들이 주어졌을 때,
    증거 e와 일관된 P의 엔트리들의 합을 리턴함."""
    if not variables:
        return P[e]
    Y, rest = variables[0], variables[1:]
    return sum([enumerate_joint(rest, extend(e, Y, y), P) for y in P.values(Y)])


def extend(s, var, val):
    """딕셔너리 s를 복사하고 var에 값 val을 세팅하여 확장하여 그 복사본을 리턴함."""
    return {**s, var: val}

def enumerate_joint_ask(X, e, P):
    """결합 확률 분포 P에서 {var:val} 관측들(증거들) e가 주어졌을 때,
    질의 변수 X의 값들에 대한 확률 분포를 리턴함."""
    assert X not in e, "질의 변수는 증거에 포함된 변수일 수 없음"
    Q = ProbDist(X)  # X에 대한 확률 분포, 초기에는 비어 있음.
    Y = [v for v in P.variables if v != X and v not in e]  # 미관측 변수
    for xi in P.values(X):
        Q[xi] = enumerate_joint(Y, extend(e, X, xi), P)
    return Q.normalize()
# 확률 정의~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 완전 결합 확률 분포 정의
full_joint = JointProbDist(['국어', '영어', '수학', '인공지능'])
t=True; f=False
full_joint[dict(국어=t, 영어=t, 수학=t, 인공지능=t)] = 0.02
full_joint[dict(국어=t, 영어=f, 수학=t, 인공지능=t)] = 0.13
full_joint[dict(국어=f, 영어=t, 수학=t, 인공지능=t)] = 0.02
full_joint[dict(국어=f, 영어=f, 수학=t, 인공지능=t)] = 0.01
full_joint[dict(국어=t, 영어=t, 수학=f, 인공지능=t)] = 0.12
full_joint[dict(국어=t, 영어=f, 수학=f, 인공지능=t)] = 0.08
full_joint[dict(국어=f, 영어=t, 수학=f, 인공지능=t)] = 0.13
full_joint[dict(국어=f, 영어=f, 수학=f, 인공지능=t)] = 0.04
full_joint[dict(국어=t, 영어=t, 수학=t, 인공지능=f)] = 0.03
full_joint[dict(국어=t, 영어=f, 수학=t, 인공지능=f)] = 0.04 
full_joint[dict(국어=f, 영어=t, 수학=t, 인공지능=f)] = 0.07
full_joint[dict(국어=f, 영어=f, 수학=t, 인공지능=f)] = 0.08
full_joint[dict(국어=t, 영어=t, 수학=f, 인공지능=f)] = 0.1
full_joint[dict(국어=t, 영어=f, 수학=f, 인공지능=f)] = 0.03
full_joint[dict(국어=f, 영어=t, 수학=f, 인공지능=f)] = 0.06
full_joint[dict(국어=f, 영어=f, 수학=f, 인공지능=f)] = 0.04


# 쿼리문1
# P(인공지능=True)인 경우 조건부 확률
evidence = dict(인공지능=t)
variables = ['국어','영어', '수학']     # 증거에 포함되지 않은 변수들
ans1 = enumerate_joint(variables, evidence, full_joint)
print('인공지능 시험을 잘봤을 확률(국영수는 고려X):', ans1)

# 쿼리문2
# P(국어=f, 인공지능=t)                         # 여기서부터 다시 시작하기~~~~~~~~~~~~~~~~~~~~~~~~~
evidence = dict(국어=f,  인공지능=t)
variables = [''] # 증거에 포함되지 않은 변수들
ans2 = enumerate_joint(variables, evidence, full_joint)
print('캐피티가참, 투쓰에이크가참일때의 확률', ans2)

# 이를 활용한 조건부확률 구하기
# P(Cavity=True | Toothache=True) = P(Cavity=True and Toothache=True) / P(Toothache=True)
# ans2/ans1

# 쿼리문3
# P(Cavity | Toothache=True)
# query_variable = 'Cavity'
# evidence = dict(Toothache=True)
# ans = enumerate_joint_ask(query_variable, evidence, full_joint)
# (ans[True], ans[False])




# 여기서부터 구현을 시작한다(국어, 영어, 수학, 인공지능 시험을 잘볼 확률을 비교)
# 잘보면 True, 못보면 False를 의미한다


# 완전결합 확률의 합은 1.0이 되게 한다.
#print(sum([0.02,0.13, 0.02, 0.01, 0.12, 0.08, 0.13, 0.04, 0.03, 0.04, 0.07, 0.08, 0.1, 0.03, 0.06, 0.04]))