# SimCLE

Learning Sentence Embeddings via translation datasets

It is a implemention of pytorch vision

Train the sentence representations with your own translation dataset:
  The model file: model.py
  The trianing file: simcle_contrastive.py
  Setup instruction: ./main.sh
  
  The hyperparameter in main.sh: \n
                    --save_model_path The path you want to save your pre-trained model. \n
                    --batch_size \n
                    --eval_step You evaluate your model every steps in training. \n
                    --queue_len The size of queueu containing negative samples, which is used to expand the comparaing objectives in contrastive learning. \n
                    --lr learning rate \n
                    --epochs \n
                    --task The pre-trainging task (default "plus" is represent the task combining contrastive objective and self-contrasting) \n
                    --zh_pmp The Chinese pre-training weight file （The pre-training model is based on the model "hfl/chinese-roberta-wwm-ext-large", and the "zh_pmp" file will update the weight of the based model.） \n
                    --eval_name The evaluating benchmark used in evaluation task. \n
                    --temp The temperature used in contrastive learning \n
                    --dropout \n
                    --zh_data_path The Chinese side path of parallel translation dataset. \n
                    --en_data_path The English side path of parallel translation dataset. \n
       
   
                    
                    
