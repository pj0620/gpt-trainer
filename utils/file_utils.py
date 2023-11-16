import os
import os
import sys

root = os.path.dirname(__file__)
root = os.path.dirname(root)
sys.path.append(root)


def get_config(problem_name):
    """
    return configuration json files for Problem
    user can customize only parts of configuration json files, other files will be left for default
    Args:
        problem_name: customized configuration name under Problems/

    Returns:
        path problem config, input images/text, solutions: [problem_config_path, problem_statements_path, solutions_path]
    """
    config_dir = os.path.join(root, "Problems", problem_name)
    default_config_dir = os.path.join(root, "Problems", "PhotoCircuit")

    config_files = [
        "problem_config.json",
        "problem_statements",
        "solutions"
    ]

    config_paths = []

    for config_file in config_files:
        problem_config_path = os.path.join(config_dir, config_file)
        default_config_path = os.path.join(default_config_dir, config_file)

        if os.path.exists(problem_config_path):
            config_paths.append(problem_config_path)
        else:
            config_paths.append(default_config_path)

    return tuple(config_paths)
