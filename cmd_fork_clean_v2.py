import random
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np

# ===============================
# CONFIGURATION
# ===============================

RR_VALUES = [10, 50, 200, 1000]
RB_VALUES = ["Fragile", "Moderate", "Secure"]
DC_VALUES = ["Low", "Moderate", "High"]
IRREVERSIBILITY_VALUES = ["Reversible", "Partially Reversible", "Irreversible"]

AXES = {
    1: "Risk Posture",
    2: "Adaptation Threshold",
    3: "Governance Geometry"
}

PROBABILITY_STATEMENT = "70% favorable outcome / 30% meaningful negative outcome."

# ===============================
# DATA STRUCTURES
# ===============================

@dataclass
class ModifierState:
    rr: Optional[int] = None
    rb: Optional[str] = None
    dc: Optional[str] = None
    irreversibility: Optional[str] = None
    axis: Optional[int] = None

    def active_layers(self):
        layers = []
        if self.rr is not None:
            layers.append("RR")
        if self.rb is not None:
            layers.append("RB")
        if self.dc is not None:
            layers.append("DC")
        if self.irreversibility is not None:
            layers.append("IRREVERSIBILITY")
        return layers

    def to_dict(self):
        return {
            "axis": self.axis,
            "rr": self.rr,
            "rb": self.rb,
            "dc": self.dc,
            "irreversibility": self.irreversibility,
            "layer_count": len(self.active_layers())
        }

# ===============================
# PERSONALITY GENERATOR
# ===============================

class PersonalityGenerator:
    """
    Generates diverse decision-making personalities using Dirichlet distributions.
    Each personality has biases toward A/B/C decisions.
    """
    
    @staticmethod
    def generate_random_personality(alpha_concentration=1.0):
        """
        Generate A/B/C probability weights from Dirichlet distribution.
        
        alpha_concentration:
        - 1.0 = uniform (no bias)
        - <1.0 = extreme personalities (spiky)
        - >1.0 = moderate personalities (balanced)
        """
        weights = np.random.dirichlet([alpha_concentration] * 3)
        return {
            "A": weights[0],
            "B": weights[1],
            "C": weights[2],
            "type": "random_dirichlet"
        }
    
    @staticmethod
    def generate_personality_pool(n=100, alpha=1.0):
        """Generate a pool of N diverse personalities"""
        return [PersonalityGenerator.generate_random_personality(alpha) for _ in range(n)]

# ===============================
# AUTO DECISION ENGINE (CLEANED)
# ===============================

class AutoDecisionEngine:
    """
    OPERATOR PROFILE REMOVED - Pure random decision simulation.
    
    Modes:
    - "pure_random": Uniform 33/33/33 split
    - "personality": Use pre-generated personality weights
    - "dirichlet": Generate fresh personality each run
    """
    
    def __init__(self, mode="pure_random", personality=None):
        self.mode = mode
        self.personality = personality
        
        if mode == "personality" and personality is None:
            # Generate random personality if none provided
            self.personality = PersonalityGenerator.generate_random_personality()
    
    def decide(self, state: ModifierState) -> str:
        if self.mode == "pure_random":
            # Completely unbiased
            return random.choice(["A", "B", "C"])
        
        elif self.mode == "personality":
            # Use personality weights
            r = random.random()
            if r < self.personality["A"]:
                return "A"
            elif r < self.personality["A"] + self.personality["B"]:
                return "B"
            else:
                return "C"
        
        elif self.mode == "dirichlet":
            # Fresh personality each decision (extreme randomness)
            weights = np.random.dirichlet([1.0, 1.0, 1.0])
            r = random.random()
            if r < weights[0]:
                return "A"
            elif r < weights[0] + weights[1]:
                return "B"
            else:
                return "C"
        
        else:
            # Fallback to pure random
            return random.choice(["A", "B", "C"])


# ===============================
# SUCCESS SIMULATOR
# ===============================

class SuccessSimulator:
    """
    Simulates success/failure outcomes based on decision + modifier state.
    This adds the "success" dimension missing from original data.
    """
    
    @staticmethod
    def calculate_success(decision: str, state: ModifierState) -> float:
        """
        Returns success probability (0.0 to 1.0) based on decision and modifiers.
        
        Heuristic model:
        - Base success varies by decision type
        - Modified by environmental factors
        - Layer complexity applies penalty
        """
        
        # Base success rates
        base = {
            "A": 0.65,  # Protect - safe but conservative
            "B": 0.70,  # Risk - higher reward, higher variance
            "C": 0.60   # Context - deliberation cost
        }
        
        success_prob = base[decision]
        
        # RR modifier (responsibility radius stress)
        if state.rr is not None:
            if state.rr >= 1000:
                success_prob -= 0.05
            elif state.rr >= 200:
                success_prob -= 0.02
        
        # RB modifier (resource buffer)
        if state.rb == "Fragile":
            success_prob -= 0.10
        elif state.rb == "Secure":
            success_prob += 0.05
        
        # DC modifier (damage containment)
        if state.dc == "High":
            success_prob += 0.08
        elif state.dc == "Low":
            success_prob -= 0.08
        
        # Irreversibility penalty
        if state.irreversibility == "Irreversible":
            success_prob -= 0.10
        elif state.irreversibility == "Partially Reversible":
            success_prob -= 0.05
        
        # Layer complexity penalty
        layer_count = len(state.active_layers())
        success_prob -= 0.03 * (layer_count - 1)
        
        # Decision-specific modifiers
        if decision == "A":
            # Protect works better in fragile/irreversible
            if state.rb == "Fragile":
                success_prob += 0.12
            if state.irreversibility in ["Irreversible", "Partially Reversible"]:
                success_prob += 0.08
        
        elif decision == "B":
            # Risk works better with buffer/reversibility
            if state.rb == "Secure":
                success_prob += 0.10
            if state.irreversibility == "Reversible":
                success_prob += 0.08
            # But catastrophic in fragile + irreversible
            if state.rb == "Fragile" and state.irreversibility == "Irreversible":
                success_prob -= 0.20
        
        elif decision == "C":
            # Context helps in moderate complexity, fails at extremes
            layer_count = len(state.active_layers())
            if layer_count == 2 or layer_count == 3:
                success_prob += 0.05
            elif layer_count == 4:
                success_prob -= 0.08  # Analysis paralysis
        
        # Clamp to valid range
        success_prob = max(0.0, min(1.0, success_prob))
        
        # Actually simulate success/failure
        return 1.0 if random.random() < success_prob else 0.0


# ===============================
# ENGINE
# ===============================

class CMDForkWorkflow:

    def __init__(self, mode="pure_random", n_runs=1000, n_personalities=100, 
                 simulate_success=True, alpha_concentration=1.0):
        self.results = []
        self.n_runs = n_runs
        self.simulate_success = simulate_success
        self.mode = mode
        
        # Trust collapse scenario configuration
        self.trust_collapse_config = None
        if mode.startswith("trust_collapse"):
            if "low_dc" in mode:
                self.trust_collapse_config = {"dc_lock": "Low", "scenario": "No Containment"}
            elif "mod_dc" in mode:
                self.trust_collapse_config = {"dc_lock": "Moderate", "scenario": "Partial Containment"}
            elif "high_dc" in mode:
                self.trust_collapse_config = {"dc_lock": "High", "scenario": "Full Containment"}
        
        # Generate personality pool if needed
        if mode == "personality_pool" or mode.startswith("trust_collapse"):
            print(f"Generating {n_personalities} personalities (alpha={alpha_concentration})...")
            self.personalities = PersonalityGenerator.generate_personality_pool(
                n_personalities, alpha_concentration
            )
        else:
            self.personalities = None
        
        print(f"\nCMD-Fork v1 — CLEAN MODE ({mode}) — {n_runs} runs")
        print(f"Operator profile: REMOVED ✓")
        if self.trust_collapse_config:
            print(f"Trust Collapse Scenario: {self.trust_collapse_config['scenario']}")
            print(f"DC locked to: {self.trust_collapse_config['dc_lock']}")
        print(f"Success simulation: {'ENABLED' if simulate_success else 'DISABLED'}\n")

    def run(self):
        for run_id in range(self.n_runs):
            if run_id % 1000 == 0:
                print(f"  Run {run_id}/{self.n_runs}...")
            
            # Select personality for this run
            if self.mode == "personality_pool" or self.mode.startswith("trust_collapse"):
                personality = random.choice(self.personalities)
                decision_engine = AutoDecisionEngine(mode="personality", personality=personality)
                personality_id = self.personalities.index(personality)
            else:
                decision_engine = AutoDecisionEngine(mode=self.mode)
                personality_id = None
            
            # Random axis
            axis_id = random.choice(list(AXES.keys()))
            
            # Random RR order
            rr_order = random.sample(RR_VALUES, len(RR_VALUES))
            
            # Trust collapse scenarios: bias toward high RR and fragile RB
            if self.trust_collapse_config:
                # Weight RR toward high values (simulating systemic risk)
                rr_order = random.choices(
                    RR_VALUES, 
                    weights=[1, 2, 3, 4],  # Favor high RR
                    k=4
                )
                # Ensure at least one high RR scenario
                if 1000 not in rr_order:
                    rr_order[random.randint(0, 3)] = 1000
            
            for round_index in range(4):
                modifier = ModifierState()
                modifier.axis = axis_id
                modifier.rr = rr_order[round_index]
                
                # Trust collapse: bias toward Fragile RB (60% fragile, 30% moderate, 10% secure)
                if self.trust_collapse_config:
                    rb_weights = [0.6, 0.3, 0.1]  # Fragile, Moderate, Secure
                    modifier.rb = random.choices(RB_VALUES, weights=rb_weights, k=1)[0]
                else:
                    modifier.rb = random.choice(RB_VALUES)

                # DC assignment
                if round_index >= 2:
                    if self.trust_collapse_config:
                        # Lock DC to scenario-specific value
                        modifier.dc = self.trust_collapse_config["dc_lock"]
                    else:
                        modifier.dc = random.choice(DC_VALUES)

                # Trust collapse: higher irreversibility rate (60% vs 30%)
                irreversibility_rate = 0.6 if self.trust_collapse_config else 0.3
                if random.random() < irreversibility_rate:
                    if self.trust_collapse_config:
                        # Bias toward full irreversibility (40% irreversible, 40% partial, 20% reversible)
                        irr_weights = [0.2, 0.4, 0.4]  # Reversible, Partial, Irreversible
                        modifier.irreversibility = random.choices(
                            IRREVERSIBILITY_VALUES, weights=irr_weights, k=1
                        )[0]
                    else:
                        modifier.irreversibility = random.choice(IRREVERSIBILITY_VALUES)

                # Get decision
                decision = decision_engine.decide(modifier)
                
                # Simulate success if enabled
                if self.simulate_success:
                    success = SuccessSimulator.calculate_success(decision, modifier)
                else:
                    success = None
                
                # Record
                result = modifier.to_dict()
                result["decision"] = decision
                result["success"] = success
                result["run_id"] = run_id
                result["round"] = round_index
                result["personality_id"] = personality_id
                
                if self.trust_collapse_config:
                    result["scenario"] = self.trust_collapse_config["scenario"]
                
                if (self.mode == "personality_pool" or self.mode.startswith("trust_collapse")) and personality_id is not None:
                    result["A_bias"] = self.personalities[personality_id]["A"]
                    result["B_bias"] = self.personalities[personality_id]["B"]
                    result["C_bias"] = self.personalities[personality_id]["C"]
                
                self.results.append(result)

        print(f"\nCompleted {self.n_runs} runs — {len(self.results)} decisions recorded.\n")

    def export_csv(self, filename=None):
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"cmd_fork_CLEAN_{self.mode}_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as f:
            if not self.results:
                return
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"Exported to {filename}")
        return filename

    def summary_stats(self):
        if not self.results:
            return
        
        total = len(self.results)
        a_count = sum(1 for r in self.results if r["decision"] == "A")
        b_count = sum(1 for r in self.results if r["decision"] == "B")
        c_count = sum(1 for r in self.results if r["decision"] == "C")
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        if self.trust_collapse_config:
            print(f"SCENARIO: {self.trust_collapse_config['scenario']}")
            print(f"DC LEVEL: {self.trust_collapse_config['dc_lock']}")
            print("-"*60)
        
        print(f"Total decisions: {total}")
        print(f"A (Protect): {a_count} ({a_count/total*100:.1f}%)")
        print(f"B (Risk):    {b_count} ({b_count/total*100:.1f}%)")
        print(f"C (Context): {c_count} ({c_count/total*100:.1f}%)")
        
        # By layer count
        print("\nBY LAYER COUNT:")
        for layers in range(1, 5):
            layer_results = [r for r in self.results if r["layer_count"] == layers]
            if layer_results:
                a = sum(1 for r in layer_results if r["decision"] == "A")
                b = sum(1 for r in layer_results if r["decision"] == "B")
                c = sum(1 for r in layer_results if r["decision"] == "C")
                n = len(layer_results)
                print(f"  {layers} layers: A={a/n*100:.1f}% B={b/n*100:.1f}% C={c/n*100:.1f}%")
        
        # Success statistics if available
        if self.simulate_success:
            successes = [r for r in self.results if r["success"] is not None]
            if successes:
                overall_success = sum(r["success"] for r in successes) / len(successes)
                print(f"\nOVERALL SUCCESS RATE: {overall_success*100:.1f}%")
                
                print("\nSUCCESS BY DECISION:")
                for dec in ["A", "B", "C"]:
                    dec_results = [r for r in successes if r["decision"] == dec]
                    if dec_results:
                        dec_success = sum(r["success"] for r in dec_results) / len(dec_results)
                        print(f"  {dec}: {dec_success*100:.1f}% (n={len(dec_results)})")
                
                print("\nSUCCESS BY LAYER:")
                for layers in range(1, 5):
                    layer_results = [r for r in successes if r["layer_count"] == layers]
                    if layer_results:
                        layer_success = sum(r["success"] for r in layer_results) / len(layer_results)
                        print(f"  {layers} layers: {layer_success*100:.1f}%")
                
                # Trust collapse specific stats
                if self.trust_collapse_config:
                    print("\nTRUST COLLAPSE METRICS:")
                    
                    # Fragile RB rate
                    fragile_count = sum(1 for r in self.results if r.get("rb") == "Fragile")
                    print(f"  Fragile RB scenarios: {fragile_count/total*100:.1f}%")
                    
                    # High RR rate
                    high_rr = sum(1 for r in self.results if r.get("rr", 0) >= 200)
                    print(f"  High RR (≥200): {high_rr/total*100:.1f}%")
                    
                    # Irreversibility rate
                    irrev = sum(1 for r in self.results if r.get("irreversibility") in ["Irreversible", "Partially Reversible"])
                    total_with_irr = sum(1 for r in self.results if r.get("irreversibility") is not None)
                    if total_with_irr > 0:
                        print(f"  Irreversible/Partial: {irrev/total_with_irr*100:.1f}%")
                    
                    # Death trap scenarios (Fragile + Irreversible + Moderate/High RR)
                    death_traps = sum(1 for r in self.results 
                                     if r.get("rb") == "Fragile" 
                                     and r.get("irreversibility") in ["Irreversible", "Partially Reversible"]
                                     and r.get("rr", 0) >= 50)
                    print(f"  Death trap scenarios: {death_traps} ({death_traps/total*100:.1f}%)")
                    
                    # Success rate in death traps
                    death_trap_results = [r for r in successes
                                         if r.get("rb") == "Fragile" 
                                         and r.get("irreversibility") in ["Irreversible", "Partially Reversible"]
                                         and r.get("rr", 0) >= 50]
                    if death_trap_results:
                        death_trap_success = sum(r["success"] for r in death_trap_results) / len(death_trap_results)
                        print(f"  Death trap success rate: {death_trap_success*100:.1f}%")
        
        return {
            "total": total,
            "A_pct": a_count/total,
            "B_pct": b_count/total,
            "C_pct": c_count/total
        }


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    import sys
    
    # Default: pure random with success simulation
    mode = "pure_random"
    n_runs = 1000
    n_personalities = 100
    alpha = 1.0
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    if len(sys.argv) > 2:
        n_runs = int(sys.argv[2])
    if len(sys.argv) > 3:
        n_personalities = int(sys.argv[3])
    if len(sys.argv) > 4:
        alpha = float(sys.argv[4])
    
    print("\n" + "="*60)
    print("CMD-FORK v1 - CLEAN ENGINE")
    print("="*60)
    print("Operator profile: REMOVED")
    print(f"Mode: {mode}")
    print(f"Runs: {n_runs}")
    if mode == "personality_pool" or mode.startswith("trust_collapse"):
        print(f"Personalities: {n_personalities} (alpha={alpha})")
    if mode.startswith("trust_collapse"):
        print("\n⚠️  TRUST COLLAPSE SCENARIO ACTIVE ⚠️")
        print("- High RR bias (systemic risk)")
        print("- 60% Fragile RB (resource scarcity)")
        print("- 60% Irreversibility rate (path dependence)")
        if "low_dc" in mode:
            print("- DC: LOW (no containment capability)")
        elif "mod_dc" in mode:
            print("- DC: MODERATE (partial containment)")
        elif "high_dc" in mode:
            print("- DC: HIGH (full containment capability)")
    print("="*60 + "\n")
    
    workflow = CMDForkWorkflow(
        mode=mode, 
        n_runs=n_runs, 
        n_personalities=n_personalities,
        simulate_success=True,
        alpha_concentration=alpha
    )
    workflow.run()
    stats = workflow.summary_stats()
    filename = workflow.export_csv()
    
    print(f"\n✓ Clean simulation complete!")
    print(f"✓ Results saved to: {filename}")
    print(f"\nModes available:")
    print("  - pure_random: Uniform 33/33/33 (no bias)")
    print("  - personality_pool: 100 diverse personalities")
    print("  - dirichlet: Fresh random personality per decision")
    print("  - trust_collapse_low_dc: Crisis mode, no containment")
    print("  - trust_collapse_mod_dc: Crisis mode, partial containment")
    print("  - trust_collapse_high_dc: Crisis mode, full containment")
