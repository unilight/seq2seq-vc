import os

def get_mcd_cer(path):
    with open(path, "r") as f:
        lines = f.read().splitlines()
    for line in lines:
        if "Mean MCD" in line:
            mcd = float(line.split(":")[1].split(" ")[1])
            cer = float(line.split(":")[1].split(" ")[-1])
            return mcd, cer

for spk in ["002", "003", "008", "009", "010", "011"]:
    root_dir = f"exp/EL_PS_MALE{spk}_SP_PS_MALE{spk}_aas_vc.melmelmel.v1/results"
    all_mcds = []
    
    all_ckpt_dirs = os.listdir(root_dir)
    for ckpt_dir in all_ckpt_dirs:
        ckpt = int(ckpt_dir.split("-")[1].split("steps")[0])
        dev_eval_log_path = os.path.join(root_dir, ckpt_dir, f"EL_PS_MALE{spk}_dev", "evaluation.log")
        mcd, _ = get_mcd_cer(dev_eval_log_path)
        all_mcds.append([ckpt, mcd])

    best_ckpt = sorted(all_mcds, key=lambda x: x[1])[0][0]
    eval_eval_log_path = os.path.join(root_dir, f"checkpoint-{best_ckpt}steps", f"EL_PS_MALE{spk}_eval", "evaluation.log")
    mcd, cer = get_mcd_cer(eval_eval_log_path)

    print(spk, best_ckpt, mcd, cer)

print("===")

for spk in ["002", "003", "008", "009", "010", "011"]:
    root_dir = f"exp/EL_PS_MALE{spk}_SP_PS_MALE{spk}_aas_vc.melmelmel.v1/results"
    all_mcds = []
    
    all_ckpt_dirs = os.listdir(root_dir)
    for ckpt_dir in all_ckpt_dirs:
        ckpt = int(ckpt_dir.split("-")[1].split("steps")[0])
        dev_eval_log_path = os.path.join(root_dir, ckpt_dir, f"EL_PS_MALE{spk}_dev", "evaluation.log")
        mcd, cer = get_mcd_cer(dev_eval_log_path)
        all_mcds.append([ckpt, cer])

    best_ckpt = sorted(all_mcds, key=lambda x: x[1])[0][0]
    eval_eval_log_path = os.path.join(root_dir, f"checkpoint-{best_ckpt}steps", f"EL_PS_MALE{spk}_eval", "evaluation.log")
    mcd, cer = get_mcd_cer(eval_eval_log_path)

    print(spk, best_ckpt, mcd, cer)
