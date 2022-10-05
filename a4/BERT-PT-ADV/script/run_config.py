import json
import os
from reviewlab import TaskConfig, TrainConfig, BaselineConfig

trainconfig = TrainConfig()

taskconfigs = [
   # TaskConfig("ae", "14", "laptop"),
    TaskConfig("ae", "16", "rest"),
]

baselines = [
    #BaselineConfig("BERT", model_name_or_path="bert-base-uncased"),
    #BaselineConfig("BERT-DK", model_name_or_path="activebus/BERT-DK_[DOMAIN]"),
    BaselineConfig("BERT-PT", model_name_or_path="activebus/BERT-PT_[DOMAIN]"),
    
    #BaselineConfig("BERT_Review", model_name_or_path="activebus/BERT_Review"),
    # BaselineConfig("Bert_Review", model_name_or_path="pt_runs/pt_bert-Bert"),
    #BaselineConfig("BERT-XD_Review", model_name_or_path="activebus/BERT-XD_Review"),
    # BaselineConfig("BERT-XD_Review", model_name_or_path="pt_runs/pt_bertselect-SkipDomBert"),
    
    
]


if __name__=="__main__":
    for baseline in baselines:
        for taskconfig in taskconfigs:
            config = {}
            config.update(trainconfig.to_dict())
            config.update(taskconfig.to_dict())
            config.update(baseline.to_dict())
            config["model_name_or_path"] = config["model_name_or_path"].replace("[DOMAIN]", config["domain"])
            config_name = "_".join([config["task"], config["domain"], config["year"], config["baseline"]])
            config["output_dir"] = os.path.join("ft_runs", config_name)

            # ft_runs
            os.makedirs(config["output_dir"], exist_ok=True)
            
            with open(os.path.join(config["output_dir"], "config.json"), "w") as fw:
                json.dump(config, fw)
            
            with open("configs/" + config_name + ".json", "w") as fw:
                json.dump(config, fw)            
