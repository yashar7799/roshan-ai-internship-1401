# from .main import main_args

ARGUMENTS = dict(
    # model1=main_args
)

def get_args(model_name):
    """Get Argument Parser"""
    return ARGUMENTS[model_name]()