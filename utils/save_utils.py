import os
import pickle
import shutil


#save utils
def save_data(save_obj,save_path,token_budget=None,sample_idx=None,batch_idxs=None, config_save=False):

    # Check how many files exist in directory
    os.makedirs(save_path,exist_ok=True)


    if config_save:
        save_file_path=f"{save_path}/config.pkl"
        with open(save_file_path, 'wb') as f:
            pickle.dump(save_obj, f)

    if not config_save:
        #get number of files in directory
        if token_budget is not None and sample_idx is not None:
            save_file_path = os.path.join(save_path, f"save_file__sample_{sample_idx}_{token_budget}.pkl")
        elif token_budget is not None and batch_idxs is not None:
            save_file_path = os.path.join(save_path, f"save_file__batch_{batch_idxs}_{token_budget}.pkl")
        elif token_budget is not None:
            save_file_path = os.path.join(save_path, f"save_file_{token_budget}.pkl")
        elif batch_idxs is not None:
            save_file_path = os.path.join(save_path, f"save_file__batch_{batch_idxs}.pkl")
        else:
            save_file_path = os.path.join(save_path, "save_file.pkl")

        with open(save_file_path, 'wb') as f:
            pickle.dump(save_obj, f)

def safe_delete(dir):
    if os.path.exists(dir):
        if os.path.isfile(dir):
            os.remove(dir)
        else:
            shutil.rmtree(dir)
    else:
        print(f"File {dir} does not exist")

