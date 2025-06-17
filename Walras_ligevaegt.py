#!/usr/bin/env python3
"""
Walras Equilibrium Calculator CLI
Calculates equilibrium prices and allocations given utility functions and initial endowments.
"""

import sympy as sp
from sympy import symbols, solve, diff, simplify, sympify, latex
from InquirerPy import inquirer
from InquirerPy.validator import NumberValidator, EmptyInputValidator
import numpy as np
from typing import List, Dict, Tuple, Optional
import re

class WalrasCalculator:
    def __init__(self):
        self.utility_functions = []
        self.initial_endowments = []
        self.num_agents = 0
        self.num_goods = 0
        self.unknown_params = []
        self.has_unknowns = False
        
    def get_utility_functions(self) -> List[str]:
        """Get utility functions from user input"""
        print("\n=== UTILITY FUNCTIONS ===")
        print("Enter utility functions for each agent.")
        print("Examples: 'x1*x2', 'log(x1) + log(x2)', 'x1^0.5 * x2^0.5'")
        print("Use x1, x2, x3, ... for goods. Use standard mathematical notation.")
        
        # Get number of agents
        self.num_agents = int(inquirer.number(
            message="How many agents are there?",
            min_allowed=1,
            max_allowed=10,
            validate=NumberValidator(),
            default=2
        ).execute())
        
        utility_functions = []
        for i in range(self.num_agents):
            while True:
                try:
                    func_str = inquirer.text(
                        message=f"Enter utility function for agent {i+1}:",
                        validate=EmptyInputValidator()
                    ).execute()
                    
                    # Test if the function can be parsed
                    test_func = sympify(func_str)
                    utility_functions.append(func_str)
                    break
                except Exception as e:
                    print(f"Invalid function format: {e}")
                    print("Please use proper mathematical notation (e.g., x1*x2, log(x1), x1**0.5)")
        
        return utility_functions
    
    def check_unknown_parameters(self, utility_functions: List[str]) -> Tuple[bool, List[str]]:
        """Check if utility functions contain unknown parameters"""
        print("\n=== PARAMETER ANALYSIS ===")
        
        # Extract all symbols from utility functions
        all_symbols = set()
        for func_str in utility_functions:
            func = sympify(func_str)
            all_symbols.update(func.free_symbols)
        
        # Filter out good variables (x1, x2, x3, ...)
        good_pattern = re.compile(r'^x\d+$')
        potential_params = [str(sym) for sym in all_symbols if not good_pattern.match(str(sym))]
        
        if potential_params:
            print(f"Detected potential unknown parameters: {', '.join(potential_params)}")
            has_unknowns = inquirer.confirm(
                message="Do your utility functions contain unknown parameters (a, b, alpha, etc.)?",
                default=True
            ).execute()
        else:
            has_unknowns = inquirer.confirm(
                message="Do your utility functions contain unknown parameters?",
                default=False
            ).execute()
        
        unknown_params = []
        if has_unknowns:
            if potential_params:
                # Ask user to confirm which are actually unknown parameters
                unknown_params = inquirer.checkbox(
                    message="Select which symbols are unknown parameters:",
                    choices=potential_params,
                    validate=lambda x: len(x) > 0 or "Please select at least one parameter"
                ).execute()
            else:
                # Manual entry
                param_input = inquirer.text(
                    message="Enter unknown parameters separated by commas (e.g., a,b,alpha):",
                    validate=EmptyInputValidator()
                ).execute()
                unknown_params = [p.strip() for p in param_input.split(',')]
        
        return has_unknowns, unknown_params
    
    def get_initial_endowments(self, num_agents: int, num_goods: int) -> List[List[float]]:
        """Get initial endowments for each agent"""
        print(f"\n=== INITIAL ENDOWMENTS ===")
        print(f"Enter initial endowments for {num_agents} agents and {num_goods} goods")
        
        endowments = []
        for i in range(num_agents):
            print(f"\nAgent {i+1} endowments:")
            agent_endowment = []
            for j in range(num_goods):
                endowment = float(inquirer.number(
                    message=f"  Initial amount of good {j+1}:",
                    min_allowed=0,
                    validate=NumberValidator()
                ).execute())
                agent_endowment.append(endowment)
            endowments.append(agent_endowment)
        
        return endowments
    
    def determine_num_goods(self, utility_functions: List[str]) -> int:
        """Determine number of goods from utility functions"""
        max_good = 0
        for func_str in utility_functions:
            # Find all x variables (x1, x2, x3, ...)
            matches = re.findall(r'x(\d+)', func_str)
            if matches:
                max_good = max(max_good, max(int(m) for m in matches))
        return max_good
    
    def calculate_equilibrium(self) -> Dict:
        """Calculate Walras equilibrium"""
        print(f"\n=== CALCULATING EQUILIBRIUM ===")
        
        # Create symbols
        goods_symbols = [symbols(f'x{i+1}_{j+1}') for j in range(self.num_agents) for i in range(self.num_goods)]
        price_symbols = [symbols(f'p{i+1}') for i in range(self.num_goods)]
        param_symbols = [symbols(param) for param in self.unknown_params] if self.has_unknowns else []
        
        # Normalize prices (set p1 = 1 as numeraire)
        normalized_prices = [1] + price_symbols[1:] if self.num_goods > 1 else [1]
        
        try:
            # Parse utility functions
            utility_funcs = []
            for i, func_str in enumerate(self.utility_functions):
                # Replace xi with xi_j for agent j
                modified_func = func_str
                for k in range(self.num_goods):
                    modified_func = modified_func.replace(f'x{k+1}', f'x{k+1}_{i+1}')
                utility_funcs.append(sympify(modified_func))
            
            # Create demand functions using FOCs
            equations = []
            
            # For each agent, create FOCs and budget constraint
            for i in range(self.num_agents):
                agent_vars = [symbols(f'x{j+1}_{i+1}') for j in range(self.num_goods)]
                
                # First-order conditions: MU_i/p_i = lambda for all goods
                marginal_utilities = [diff(utility_funcs[i], var) for var in agent_vars]
                
                # Budget constraint
                income = sum(normalized_prices[j] * self.initial_endowments[i][j] for j in range(self.num_goods))
                budget = sum(normalized_prices[j] * agent_vars[j] for j in range(self.num_goods)) - income
                equations.append(budget)
                
                # FOCs (MU_i/p_i = MU_j/p_j for all i,j)
                if len(marginal_utilities) > 1:
                    for j in range(1, len(marginal_utilities)):
                        if normalized_prices[0] != 0 and normalized_prices[j] != 0:
                            foc = marginal_utilities[0]/normalized_prices[0] - marginal_utilities[j]/normalized_prices[j]
                            equations.append(foc)
            
            # Market clearing conditions
            for j in range(self.num_goods):
                market_clear = sum(symbols(f'x{j+1}_{i+1}') for i in range(self.num_agents)) - sum(self.initial_endowments[i][j] for i in range(self.num_agents))
                equations.append(market_clear)
            
            # Solve the system
            all_vars = goods_symbols + price_symbols[1:]  # p1 is normalized to 1
            
            print("Solving equilibrium system...")
            if self.has_unknowns:
                print("System contains unknown parameters - providing symbolic solution")
                solution = solve(equations, all_vars, dict=True)
            else:
                solution = solve(equations, all_vars, dict=True)
            
            return {
                'solution': solution,
                'equations': equations,
                'variables': all_vars,
                'prices': normalized_prices,
                'success': len(solution) > 0 if isinstance(solution, list) else solution is not None
            }
            
        except Exception as e:
            print(f"Error in calculation: {e}")
            return {'success': False, 'error': str(e)}
    
    def display_results(self, results: Dict):
        """Display the equilibrium results"""
        if not results['success']:
            print(f"\n❌ Could not find equilibrium solution.")
            if 'error' in results:
                print(f"Error: {results['error']}")
            return
        
        print(f"\n✅ WALRAS EQUILIBRIUM SOLUTION")
        print("=" * 50)
        
        solution = results['solution']
        
        if isinstance(solution, list):
            if len(solution) == 0:
                print("No solution found.")
                return
            elif len(solution) > 1:
                print(f"Multiple solutions found ({len(solution)}). Showing first solution:")
            solution = solution[0]
        
        # Display equilibrium prices
        print("\n📊 EQUILIBRIUM PRICES:")
        print(f"  p1 = 1 (numeraire)")
        for i in range(1, self.num_goods):
            price_var = symbols(f'p{i+1}')
            if price_var in solution:
                price_val = solution[price_var]
                print(f"  p{i+1} = {price_val}")
        
        # Display equilibrium allocations
        print(f"\n🎯 EQUILIBRIUM ALLOCATIONS:")
        for i in range(self.num_agents):
            print(f"\nAgent {i+1}:")
            for j in range(self.num_goods):
                var = symbols(f'x{j+1}_{i+1}')
                if var in solution:
                    allocation = solution[var]
                    print(f"  Good {j+1}: {allocation}")
        
        # If symbolic solution, show parameter dependencies
        if self.has_unknowns:
            print(f"\n🔢 SOLUTION DEPENDS ON PARAMETERS: {', '.join(self.unknown_params)}")
    
    def run(self):
        """Main CLI interface"""
        print("🏛️  WALRAS EQUILIBRIUM CALCULATOR")
        print("=" * 40)
        
        # Get utility functions
        self.utility_functions = self.get_utility_functions()
        
        # Determine number of goods
        self.num_goods = self.determine_num_goods(self.utility_functions)
        print(f"\nDetected {self.num_goods} goods from utility functions")
        
        # Check for unknown parameters
        self.has_unknowns, self.unknown_params = self.check_unknown_parameters(self.utility_functions)
        
        # Get initial endowments
        self.initial_endowments = self.get_initial_endowments(self.num_agents, self.num_goods)
        
        # Display summary
        print(f"\n📋 SUMMARY:")
        print(f"  Agents: {self.num_agents}")
        print(f"  Goods: {self.num_goods}")
        print(f"  Unknown parameters: {self.unknown_params if self.has_unknowns else 'None'}")
        
        # Confirm calculation
        proceed = inquirer.confirm(
            message="Proceed with equilibrium calculation?",
            default=True
        ).execute()
        
        if proceed:
            results = self.calculate_equilibrium()
            self.display_results(results)
        
        # Ask if user wants to save results or run again
        action = inquirer.select(
            message="What would you like to do?",
            choices=[
                "Run another calculation",
                "Exit"
            ]
        ).execute()
        
        if action == "Run another calculation":
            self.__init__()  # Reset state
            self.run()

def main():
    """Entry point"""
    try:
        calculator = WalrasCalculator()
        calculator.run()
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    main()