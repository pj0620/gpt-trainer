import argparse

from learning.gpt_model import GPTModel
from utils.file_utils import get_config

parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--problem', type=str, default="PhotoCircuit",
                    help="Name of problem gpt should solve defined in Problems/")
parser.add_argument('--version', type=str, default="v1.0.0",
                    help="Version of solution to the problem")
parser.add_argument('--gpt-model', type=str, default="gpt-4-vision-preview",
                    help="GPT Model")
parser.add_argument('--evaluator-model', type=str, default="gpt-3.5-turbo",
                    help="Evaluator Model")
parser.add_argument('--train-test-split', type=float, default=0.2,
                    help="Percent of data to use for testing, rest is used for training(between 0.0 and 1.0)")
args = parser.parse_args()

# ----------------------------------------
#          Init Learning Model
# ----------------------------------------
problem_config_path, problem_statements_path, solutions_path = get_config(args.problem)
learning_model = GPTModel(
    problem_config_path=problem_config_path,
    problem_statements_path=problem_statements_path,
    solutions_path=solutions_path,
    problem_name=args.problem,
    solution_version=args.version,
    gpt_model=args.gpt_model,
    evaluator_model=args.evaluator_model,
    test_size=args.train_test_split
)
learning_model.log("loaded config and created LearningModel instance")

# ----------------------------------------
#          Pre Processing
# ----------------------------------------
learning_model.pre_processing()

# ----------------------------------------
#          start training
# ----------------------------------------
learning_model.do_training()
