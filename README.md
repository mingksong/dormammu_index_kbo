## 모멘텀 수비 지수(MDI) 연구 및 구현 계획서 (개정판)
## 가제 도르마무(Dormammu) 연구 및 구현 계획서 (개정판)

### 1. 서론: '보이지 않는 가치'의 정량화

본 계획서는 야구 경기에서 기존 통계 지표들이 포착하지 못했던 **'숨겨진 수비 가치'와 '기세 변화'를 정량화**하기 위한 **모멘텀 수비 지수(Momentum Defense Index, MDI)**의 연구 및 구현 방안을 제시합니다. WPA(Win Probability Added)와 같은 기존의 승리 확률 기반 지표들은 실제로 발생한 플레이의 영향을 측정하는 데 탁월하지만, **득점으로 이어지지 않은 중요한 수비 플레이나 위기 극복 장면**이 경기의 흐름에 미친 영향을 간과하는 한계를 지닙니다.

MDI는 이러한 **WPA의 '서사적 한계'를 보완**하고, 특히 **높은 레버리지 상황(high-leverage situations)**에서 수비팀이 위기를 성공적으로 막아냈을 때 발생하는 기세 변화를 객관적인 수치로 제시합니다. 이는 전통적인 통계적 엄밀성과 한국 야구 문화의 '분위기'라는 서사적 요소를 결합하는 혁신적 시도입니다.

### 2. MDI 정의 및 핵심 개념

#### 2.1 이론적 프레임워크

MDI는 **반사실적 분석(Counterfactual Analysis)**과 **베이지안 추론(Bayesian Inference)**을 결합한 복합 지표입니다.

**핵심 구성 요소:**

1. **확장된 레버리지 인덱스 (Extended Leverage Index, eLI)**
   - 기존 LI에 추가 맥락 정보를 통합한 지표
   - eLI = LI × Context_Multiplier
   - Context_Multiplier는 이닝(후반기 가중), 시즌 시점(9월 가중), 상대팀 강도를 반영

2. **결정적 포인트 (Critical Point, C.P.)**
   - 베이지안 프레임워크 기반 계산:
   $$C.P. = \sum_{i \in S} [P(outcome_i | \theta) \times WPA(outcome_i)]$$
   
   여기서:
   - S = 성공적 공격 결과 집합
   - θ = {리그 평균, 매치업 히스토리, 최근 폼, 환경 요인}
   - P(outcome_i | θ)는 베이지안 추론으로 업데이트

3. **모멘텀 수비 (MD)**
   $$MD = (C.P. - WPA_{actual}) \times eLI_{normalized}$$
   
   eLI_normalized는 상황의 중요도를 0-1 범위로 정규화한 값

#### 2.2 수학적 정의

**상태 공간 정의:**
- 총 10,560개 상태: 24(베이스-아웃) × 22(점수차) × 10(이닝) × 2(홈/원정)

**전이 확률 행렬:**
$$P_{ij} = P(State_j | State_i, Action)$$

각 전이 확률은 다음의 계층적 모델로 추정:
$$P_{ij} = \pi_0 P_{league} + \pi_1 P_{team} + \pi_2 P_{player} + \epsilon$$

여기서 π는 신뢰도 가중치, ε는 잔차항

### 3. 구현 아키텍처

#### 3.1 데이터 파이프라인

```python
class MDIDataPipeline:
    def __init__(self, db_config, api_config):
        self.db = DatabaseConnection(db_config)
        self.api = APIClient(api_config)
        self.cache = RedisCache()
        
    def process_game_data(self, game_id):
        # 1. 실시간 데이터 수집
        raw_data = self.api.get_play_by_play(game_id)
        
        # 2. 상태 변환 및 보강
        enriched_data = self.enrich_with_context(raw_data)
        
        # 3. WE/WPA 계산
        probability_data = self.calculate_probabilities(enriched_data)
        
        # 4. 캐싱 및 저장
        self.cache.set(f"game:{game_id}", probability_data)
        self.db.bulk_insert('play_data', probability_data)
        
        return probability_data
```

#### 3.2 베이지안 C.P. 계산 엔진

```python
class BayesianCPCalculator:
    def __init__(self, prior_seasons=3):
        self.prior_data = self.load_historical_data(prior_seasons)
        self.transition_model = self.build_transition_model()
        
    def calculate_cp(self, game_state, context):
        # 1. Prior 설정 (리그 평균)
        prior = self.get_league_prior(game_state)
        
        # 2. Likelihood 계산 (매치업 데이터)
        likelihood = self.calculate_likelihood(
            context['batter'], 
            context['pitcher'],
            context['recent_form']
        )
        
        # 3. Posterior 업데이트
        posterior = self.bayesian_update(prior, likelihood)
        
        # 4. C.P. 계산
        cp = 0
        for outcome, prob in posterior.items():
            if outcome in POSITIVE_OUTCOMES:
                wpa_value = self.get_wpa_value(game_state, outcome)
                cp += prob * wpa_value
                
        # 5. 환경 요인 조정
        cp *= self.get_environmental_adjustment(context)
        
        return cp
    
    def bayesian_update(self, prior, likelihood):
        """베이지안 정리를 사용한 확률 업데이트"""
        evidence = sum(prior[o] * likelihood[o] for o in prior)
        posterior = {}
        
        for outcome in prior:
            posterior[outcome] = (prior[outcome] * likelihood[outcome]) / evidence
            
        return posterior
```

#### 3.3 위기 상황 식별 알고리즘

```python
class CriticalSituationIdentifier:
    def __init__(self, threshold_config):
        self.base_li_threshold = threshold_config['base_li']  # 1.5
        self.dynamic_threshold = threshold_config['dynamic']  # True
        
    def is_critical(self, game_state):
        # 1. 기본 LI 체크
        if game_state['leverage_index'] < self.base_li_threshold:
            return False
            
        # 2. 상황별 추가 조건
        conditions = [
            self._check_scoring_position(game_state),
            self._check_game_context(game_state),
            self._check_momentum_context(game_state)
        ]
        
        # 3. 가중 점수 계산
        criticality_score = self._calculate_criticality_score(
            game_state, 
            conditions
        )
        
        # 4. 동적 임계값 적용
        if self.dynamic_threshold:
            threshold = self._get_dynamic_threshold(game_state)
        else:
            threshold = 0.7
            
        return criticality_score >= threshold
    
    def _calculate_criticality_score(self, game_state, conditions):
        """다차원 위기도 점수 계산"""
        weights = {
            'leverage': 0.4,
            'scoring_position': 0.25,
            'game_phase': 0.2,
            'score_differential': 0.15
        }
        
        score = (
            weights['leverage'] * min(game_state['leverage_index'] / 3, 1) +
            weights['scoring_position'] * conditions[0] +
            weights['game_phase'] * self._get_game_phase_weight(game_state) +
            weights['score_differential'] * self._get_score_weight(game_state)
        )
        
        return score
```

### 4. 통계적 검증 프레임워크

#### 4.1 신뢰성 검증

```python
class MDIValidator:
    def __init__(self, data_manager):
        self.dm = data_manager
        
    def validate_reliability(self, team_id, season):
        """Split-half 신뢰도 및 내적 일관성 검증"""
        
        # 1. Split-half reliability
        first_half = self.calculate_mdi(team_id, season, games='first_half')
        second_half = self.calculate_mdi(team_id, season, games='second_half')
        
        split_half_r = pearsonr(first_half, second_half)[0]
        
        # 2. Spearman-Brown 보정
        reliability = (2 * split_half_r) / (1 + split_half_r)
        
        # 3. Cronbach's Alpha (게임 단위)
        game_mdi_values = self.get_game_level_mdi(team_id, season)
        alpha = self.calculate_cronbach_alpha(game_mdi_values)
        
        # 4. Bootstrap 신뢰구간
        ci_lower, ci_upper = self.bootstrap_confidence_interval(
            team_id, season, n_iterations=10000
        )
        
        return {
            'split_half_reliability': reliability,
            'cronbach_alpha': alpha,
            'confidence_interval': (ci_lower, ci_upper),
            'sample_size': len(game_mdi_values)
        }
```

#### 4.2 타당성 검증

```python
def validate_mdi_validity(self):
    """구성 타당성 및 준거 타당성 검증"""
    
    # 1. 수렴 타당성: 관련 지표들과의 상관관계
    convergent_validity = {
        'team_era': self.correlate_with_era(),
        'bullpen_era': self.correlate_with_bullpen_era(),
        'defensive_efficiency': self.correlate_with_der(),
        'high_leverage_ops_against': self.correlate_with_hl_ops()
    }
    
    # 2. 판별 타당성: 무관한 지표들과의 낮은 상관관계
    discriminant_validity = {
        'team_batting_average': self.correlate_with_team_ba(),
        'stolen_base_pct': self.correlate_with_sb_pct()
    }
    
    # 3. 예측 타당성: 미래 성과 예측력
    predictive_validity = self.test_predictive_power()
    
    return {
        'convergent': convergent_validity,
        'discriminant': discriminant_validity,
        'predictive': predictive_validity
    }
```

### 5. MDI 정규화 및 조정

#### 5.1 다층 정규화 시스템

```python
class MDINormalizer:
    def normalize(self, raw_mdi_data):
        # 1단계: 기회 조정
        mdi_per_opportunity = raw_mdi_data['total_md'] / raw_mdi_data['critical_situations']
        
        # 2단계: 환경 요인 조정
        park_adjusted = self.apply_park_factors(mdi_per_opportunity)
        opponent_adjusted = self.apply_opponent_quality(park_adjusted)
        
        # 3단계: 리그 정규화 (MDI+)
        league_average = self.calculate_league_average()
        mdi_plus = (opponent_adjusted / league_average) * 100
        
        # 4단계: 안정화 조정 (작은 샘플 보정)
        stabilized_mdi = self.apply_empirical_bayes(
            mdi_plus, 
            raw_mdi_data['critical_situations']
        )
        
        return {
            'mdi_raw': raw_mdi_data['total_md'],
            'mdi_rate': mdi_per_opportunity,
            'mdi_adjusted': opponent_adjusted,
            'mdi_plus': stabilized_mdi,
            'components': {
                'park_factor': park_adjusted / mdi_per_opportunity,
                'opponent_quality': opponent_adjusted / park_adjusted,
                'stabilization_factor': stabilized_mdi / mdi_plus
            }
        }
```

### 6. 개인 선수 기여도 모델

#### 6.1 계층적 배분 시스템

```python
class PlayerMDIAllocator:
    def allocate_mdi(self, play_data):
        play_type = play_data['result_type']
        
        # 1. 기본 배분 규칙
        base_allocation = self.get_base_allocation_rules(play_type)
        
        # 2. 난이도 조정 (Statcast 데이터 활용)
        if self.has_tracking_data(play_data):
            difficulty_adjustment = self.calculate_play_difficulty(play_data)
            base_allocation = self.adjust_for_difficulty(
                base_allocation, 
                difficulty_adjustment
            )
        
        # 3. 시너지 효과 고려
        synergy_bonus = self.calculate_synergy_effects(play_data)
        
        # 4. 최종 배분
        final_allocation = {}
        for player, share in base_allocation.items():
            final_allocation[player] = share * (1 + synergy_bonus.get(player, 0))
            
        # 정규화 (합이 1이 되도록)
        total = sum(final_allocation.values())
        return {p: v/total for p, v in final_allocation.items()}
```

### 7. 실시간 분석 시스템

#### 7.1 스트리밍 아키텍처

```python
class RealTimeMDIProcessor:
    def __init__(self):
        self.stream_processor = KafkaStreamProcessor()
        self.state_manager = GameStateManager()
        self.cache = InMemoryCache()
        
    async def process_play_stream(self, game_id):
        """실시간 플레이 데이터 처리"""
        async for play in self.stream_processor.consume(f"game_{game_id}"):
            # 1. 상태 업데이트
            current_state = self.state_manager.update(play)
            
            # 2. 위기 상황 체크
            if self.is_critical_situation(current_state):
                # 3. C.P. 계산 (캐시 활용)
                cp = await self.calculate_cp_async(current_state)
                
                # 4. 플레이 결과 대기
                result = await self.wait_for_play_result()
                
                # 5. MD 계산 및 브로드캐스트
                md_value = self.calculate_md(cp, result.wpa)
                await self.broadcast_md_update(game_id, md_value)
```

### 8. 시각화 및 리포팅

#### 8.1 대시보드 컴포넌트

```javascript
// React 컴포넌트 예시
const MDIDashboard = ({ gameId }) => {
    const [mdiData, setMdiData] = useState(null);
    const [liveUpdates, setLiveUpdates] = useState([]);
    
    useEffect(() => {
        // WebSocket 연결로 실시간 업데이트
        const ws = new WebSocket(`ws://api/mdi/live/${gameId}`);
        
        ws.onmessage = (event) => {
            const update = JSON.parse(event.data);
            setLiveUpdates(prev => [...prev, update]);
            updateMDIChart(update);
        };
        
        return () => ws.close();
    }, [gameId]);
    
    return (
        <div className="mdi-dashboard">
            <MDIHeatmap data={mdiData?.heatmap} />
            <CumulativeMDIChart updates={liveUpdates} />
            <MomentumSwingIndicator critical={mdiData?.criticalMoments} />
            <PlayerContributionBreakdown players={mdiData?.playerShares} />
        </div>
    );
};
```

### 9. 실증 연구 계획

#### 9.1 파일럿 스터디 (3개월)

**목표:** KBO 2024 시즌 데이터로 MDI 모델 검증

**방법론:**
1. 10개 구단 전체 경기 데이터 수집
2. MDI 계산 및 기존 지표와 비교
3. 중계진/전문가 평가와 MDI 순위 비교
4. 팬 설문조사를 통한 체감 모멘텀과 비교

**성공 지표:**
- 신뢰도 계수 > 0.7
- 전문가 평가와의 상관계수 > 0.6
- 피타고라스 잔차 설명력 R² > 0.3

#### 9.2 본 연구 (6개월)

**확장 분석:**
1. 3년간 데이터로 MDI 안정성 검증
2. 포스트시즌 MDI와 시리즈 승패 관계
3. 선수별 MDI와 계약 가치 상관관계
4. 국제 대회 적용 가능성 검토

### 10. 한계점 및 보완 방안

#### 10.1 방법론적 한계

1. **인과관계 vs 상관관계**
   - MDI는 인과적 효과가 아닌 서술적 지표
   - 보완: 준실험적 설계로 인과 효과 추정

2. **샘플 크기 문제**
   - 고레버리지 상황의 희소성
   - 보완: 베이지안 수축 추정량 사용

3. **맥락 의존성**
   - 팀/리그별 플레이 스타일 차이
   - 보완: 계층적 모델링으로 팀 효과 분리

#### 10.2 실무적 제약

1. **계산 복잡도**
   - 실시간 처리 시 지연 가능성
   - 보완: 사전 계산 테이블 및 근사 알고리즘

2. **해석의 어려움**
   - 일반 팬들의 이해도 문제
   - 보완: 직관적 시각화 및 스토리텔링

### 11. 결론

MDI는 야구의 통계적 분석과 서사적 이해를 연결하는 혁신적 지표입니다. 엄격한 통계적 방법론과 실시간 데이터 처리 기술을 결합하여, 기존 지표들이 놓쳤던 '모멘텀'의 가치를 정량화합니다. 이는 단순한 학술적 연구를 넘어, 방송 중계, 팀 전략 수립, 팬 경험 향상에 실질적으로 기여할 수 있는 도구가 될 것입니다.
