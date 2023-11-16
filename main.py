import argparse

from learning.learning_model import LearningModel
from utils.file_utils import get_config

parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--problem', type=str, default="PhotoCircuit",
                    help="Name of problem gpt should solve defined in Problems/")
parser.add_argument('--version', type=str, default="v1.0.0",
                    help="Version of solution to the problem")
parser.add_argument('--model', type=str, default="gpt-4-vision-preview",
                    help="GPT Model")
args = parser.parse_args()

# ----------------------------------------
#          Init Learning Model
# ----------------------------------------
problem_config_path, problem_statements_path, solutions_path = get_config(args.problem)
learning_model = LearningModel(
    problem_config_path=problem_config_path,
    problem_statements_path=problem_statements_path,
    solutions_path=solutions_path,
    problem_name=args.problem,
    solution_version=args.version,
    model_type=args.model
)


learning_model.log("loaded config and created LearningModel instance")
