import torch
from collections import OrderedDict

def load_dino_model(model, checkpoint_path, verbose=False):
    """
    Args:
        model (a torchvision model): Initial model to be filled.
        checkpoint_path (path): path where the pretrained model checkpoint is stored.

    Returns:
        the filled model.
    """
    checkpoint = torch.load(checkpoint_path)
    student_state_dict = checkpoint['student']
    new_student_state_dict = OrderedDict()

    if verbose:
        print('=== Initial model state dict keys ===')
        print(model.state_dict().keys())

    for key in model.state_dict().keys():
        
        if 'module.backbone.' + key in student_state_dict.keys():
            new_student_state_dict[key] = student_state_dict['module.backbone.' + key]
            if verbose:
                print('Parameter', key, 'taken from the pretrained model')
        else:    
            new_student_state_dict[key] = model.state_dict()[key]
            if verbose:
                print('Parameter', key, 'taken from the random init')

    model.load_state_dict(new_student_state_dict)

    return model