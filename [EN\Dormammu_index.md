## Momentum Defense Index (MDI) Research and Implementation Plan (Revised)
## Project Dormammu Research and Implementation Plan (Revised)

### 1. Introduction: Quantifying 'Invisible Value'

This plan presents the research and implementation approach for the **Momentum Defense Index (MDI)**, designed to **quantify the 'hidden defensive value' and 'momentum shifts'** that existing statistical metrics have failed to capture in baseball games. While existing win probability-based metrics like WPA (Win Probability Added) excel at measuring the impact of actual plays, they have limitations in recognizing the influence on game flow from **crucial defensive plays or crisis management that didn't result in runs**.

MDI complements these **'narrative limitations' of WPA**, providing objective quantification of momentum shifts that occur when defensive teams successfully navigate **high-leverage situations**. This represents an innovative attempt to combine traditional statistical rigor with the narrative element of 'atmosphere' in Korean baseball culture.

### 2. MDI Definition and Core Concepts

#### 2.1 Theoretical Framework

MDI is a composite metric combining **Counterfactual Analysis** and **Bayesian Inference**.

**Core Components:**

1. **Extended Leverage Index (eLI)**
   - An enhanced metric integrating additional contextual information into traditional LI
   - eLI = LI × Context_Multiplier
   - Context_Multiplier reflects inning (late-game weighting), season timing (September weighting), and opponent strength

2. **Critical Point (C.P.)**
   - Calculated using Bayesian framework:
   $$C.P. = \sum_{i \in S} [P(outcome_i | \theta) \times WPA(outcome_i)]$$
   
   Where:
   - S = Set of successful offensive outcomes
   - θ = {league average, matchup history, recent form, environmental factors}
   - P(outcome_i | θ) is updated through Bayesian inference

3. **Momentum Defense (MD)**
   $$MD = (C.P. - WPA_{actual}) \times eLI_{normalized}$$
   
   eLI_normalized normalizes situation importance to 0-1 range

#### 2.2 Mathematical Definition

**State Space Definition:**
- Total 10,560 states: 24(base-out) × 22(score differential) × 10(inning) × 2(home/away)

**Transition Probability Matrix:**
$$P_{ij} = P(State_j | State_i, Action)$$

Each transition probability is estimated using hierarchical model:
$$P_{ij} = \pi_0 P_{league} + \pi_1 P_{team} + \pi_2 P_{player} + \epsilon$$

Where π represents confidence weights and ε is residual

### 3. Implementation Architecture

#### 3.1 Data Pipeline

```python
class MDIDataPipeline:
    def __init__(self, db_config, api_config):
        self.db = DatabaseConnection(db_config)
        self.api = APIClient(api_config)
        self.cache = RedisCache()
        
    def process_game_data(self, game_id):
        # 1. Real-time data collection
        raw_data = self.api.get_play_by_play(game_id)
        
        # 2. State transformation and enrichment
        enriched_data = self.enrich_with_context(raw_data)
        
        # 3. WE/WPA calculation
        probability_data = self.calculate_probabilities(enriched_data)
        
        # 4. Caching and storage
        self.cache.set(f"game:{game_id}", probability_data)
        self.db.bulk_insert('play_data', probability_data)
        
        return probability_data
```

#### 3.2 Bayesian C.P. Calculation Engine

```python
class BayesianCPCalculator:
    def __init__(self, prior_seasons=3):
        self.prior_data = self.load_historical_data(prior_seasons)
        self.transition_model = self.build_transition_model()
        
    def calculate_cp(self, game_state, context):
        # 1. Set prior (league average)
        prior = self.get_league_prior(game_state)
        
        # 2. Calculate likelihood (matchup data)
        likelihood = self.calculate_likelihood(
            context['batter'], 
            context['pitcher'],
            context['recent_form']
        )
        
        # 3. Update posterior
        posterior = self.bayesian_update(prior, likelihood)
        
        # 4. Calculate C.P.
        cp = 0
        for outcome, prob in posterior.items():
            if outcome in POSITIVE_OUTCOMES:
                wpa_value = self.get_wpa_value(game_state, outcome)
                cp += prob * wpa_value
                
        # 5. Environmental factor adjustment
        cp *= self.get_environmental_adjustment(context)
        
        return cp
    
    def bayesian_update(self, prior, likelihood):
        """Probability update using Bayes' theorem"""
        evidence = sum(prior[o] * likelihood[o] for o in prior)
        posterior = {}
        
        for outcome in prior:
            posterior[outcome] = (prior[outcome] * likelihood[outcome]) / evidence
            
        return posterior
```

#### 3.3 Critical Situation Identification Algorithm

```python
class CriticalSituationIdentifier:
    def __init__(self, threshold_config):
        self.base_li_threshold = threshold_config['base_li']  # 1.5
        self.dynamic_threshold = threshold_config['dynamic']  # True
        
    def is_critical(self, game_state):
        # 1. Basic LI check
        if game_state['leverage_index'] < self.base_li_threshold:
            return False
            
        # 2. Additional situational conditions
        conditions = [
            self._check_scoring_position(game_state),
            self._check_game_context(game_state),
            self._check_momentum_context(game_state)
        ]
        
        # 3. Calculate weighted score
        criticality_score = self._calculate_criticality_score(
            game_state, 
            conditions
        )
        
        # 4. Apply dynamic threshold
        if self.dynamic_threshold:
            threshold = self._get_dynamic_threshold(game_state)
        else:
            threshold = 0.7
            
        return criticality_score >= threshold
    
    def _calculate_criticality_score(self, game_state, conditions):
        """Calculate multi-dimensional criticality score"""
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

### 4. Statistical Validation Framework

#### 4.1 Reliability Validation

```python
class MDIValidator:
    def __init__(self, data_manager):
        self.dm = data_manager
        
    def validate_reliability(self, team_id, season):
        """Split-half reliability and internal consistency validation"""
        
        # 1. Split-half reliability
        first_half = self.calculate_mdi(team_id, season, games='first_half')
        second_half = self.calculate_mdi(team_id, season, games='second_half')
        
        split_half_r = pearsonr(first_half, second_half)[0]
        
        # 2. Spearman-Brown correction
        reliability = (2 * split_half_r) / (1 + split_half_r)
        
        # 3. Cronbach's Alpha (game level)
        game_mdi_values = self.get_game_level_mdi(team_id, season)
        alpha = self.calculate_cronbach_alpha(game_mdi_values)
        
        # 4. Bootstrap confidence interval
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

#### 4.2 Validity Validation

```python
def validate_mdi_validity(self):
    """Construct validity and criterion validity validation"""
    
    # 1. Convergent validity: correlations with related metrics
    convergent_validity = {
        'team_era': self.correlate_with_era(),
        'bullpen_era': self.correlate_with_bullpen_era(),
        'defensive_efficiency': self.correlate_with_der(),
        'high_leverage_ops_against': self.correlate_with_hl_ops()
    }
    
    # 2. Discriminant validity: low correlations with unrelated metrics
    discriminant_validity = {
        'team_batting_average': self.correlate_with_team_ba(),
        'stolen_base_pct': self.correlate_with_sb_pct()
    }
    
    # 3. Predictive validity: future performance prediction
    predictive_validity = self.test_predictive_power()
    
    return {
        'convergent': convergent_validity,
        'discriminant': discriminant_validity,
        'predictive': predictive_validity
    }
```

### 5. MDI Normalization and Adjustment

#### 5.1 Multi-layer Normalization System

```python
class MDINormalizer:
    def normalize(self, raw_mdi_data):
        # Step 1: Opportunity adjustment
        mdi_per_opportunity = raw_mdi_data['total_md'] / raw_mdi_data['critical_situations']
        
        # Step 2: Environmental factor adjustment
        park_adjusted = self.apply_park_factors(mdi_per_opportunity)
        opponent_adjusted = self.apply_opponent_quality(park_adjusted)
        
        # Step 3: League normalization (MDI+)
        league_average = self.calculate_league_average()
        mdi_plus = (opponent_adjusted / league_average) * 100
        
        # Step 4: Stabilization adjustment (small sample correction)
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

### 6. Individual Player Contribution Model

#### 6.1 Hierarchical Allocation System

```python
class PlayerMDIAllocator:
    def allocate_mdi(self, play_data):
        play_type = play_data['result_type']
        
        # 1. Base allocation rules
        base_allocation = self.get_base_allocation_rules(play_type)
        
        # 2. Difficulty adjustment (using Statcast data)
        if self.has_tracking_data(play_data):
            difficulty_adjustment = self.calculate_play_difficulty(play_data)
            base_allocation = self.adjust_for_difficulty(
                base_allocation, 
                difficulty_adjustment
            )
        
        # 3. Consider synergy effects
        synergy_bonus = self.calculate_synergy_effects(play_data)
        
        # 4. Final allocation
        final_allocation = {}
        for player, share in base_allocation.items():
            final_allocation[player] = share * (1 + synergy_bonus.get(player, 0))
            
        # Normalize (sum to 1)
        total = sum(final_allocation.values())
        return {p: v/total for p, v in final_allocation.items()}
```

### 7. Real-time Analysis System

#### 7.1 Streaming Architecture

```python
class RealTimeMDIProcessor:
    def __init__(self):
        self.stream_processor = KafkaStreamProcessor()
        self.state_manager = GameStateManager()
        self.cache = InMemoryCache()
        
    async def process_play_stream(self, game_id):
        """Process real-time play data"""
        async for play in self.stream_processor.consume(f"game_{game_id}"):
            # 1. Update state
            current_state = self.state_manager.update(play)
            
            # 2. Check critical situation
            if self.is_critical_situation(current_state):
                # 3. Calculate C.P. (with cache)
                cp = await self.calculate_cp_async(current_state)
                
                # 4. Wait for play result
                result = await self.wait_for_play_result()
                
                # 5. Calculate MD and broadcast
                md_value = self.calculate_md(cp, result.wpa)
                await self.broadcast_md_update(game_id, md_value)
```

### 8. Visualization and Reporting

#### 8.1 Dashboard Components

```javascript
// React component example
const MDIDashboard = ({ gameId }) => {
    const [mdiData, setMdiData] = useState(null);
    const [liveUpdates, setLiveUpdates] = useState([]);
    
    useEffect(() => {
        // WebSocket connection for real-time updates
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

### 9. Empirical Research Plan

#### 9.1 Pilot Study (3 months)

**Objective:** Validate MDI model with KBO 2024 season data

**Methodology:**
1. Collect all game data from 10 KBO teams
2. Calculate MDI and compare with existing metrics
3. Compare MDI rankings with broadcaster/expert evaluations
4. Compare with perceived momentum through fan surveys

**Success Metrics:**
- Reliability coefficient > 0.7
- Correlation with expert evaluation > 0.6
- Pythagorean residual explanatory power R² > 0.3

#### 9.2 Main Study (6 months)

**Extended Analysis:**
1. Verify MDI stability with 3 years of data
2. Relationship between postseason MDI and series outcomes
3. Correlation between player MDI and contract value
4. Review applicability to international competitions

### 10. Limitations and Mitigation Strategies

#### 10.1 Methodological Limitations

1. **Causation vs Correlation**
   - MDI is a descriptive, not causal metric
   - Mitigation: Estimate causal effects through quasi-experimental design

2. **Sample Size Issues**
   - Rarity of high-leverage situations
   - Mitigation: Use Bayesian shrinkage estimators

3. **Context Dependency**
   - Team/league play style differences
   - Mitigation: Separate team effects through hierarchical modeling

#### 10.2 Practical Constraints

1. **Computational Complexity**
   - Potential delays in real-time processing
   - Mitigation: Pre-calculated tables and approximation algorithms

2. **Interpretation Difficulty**
   - General fan comprehension issues
   - Mitigation: Intuitive visualization and storytelling

### 11. Conclusion

MDI is an innovative metric that bridges statistical analysis and narrative understanding in baseball. By combining rigorous statistical methodology with real-time data processing technology, it quantifies the value of 'momentum' that existing metrics have missed. This goes beyond academic research to become a practical tool that can contribute to broadcasting, team strategy development, and enhanced fan experience.
