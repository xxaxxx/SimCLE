# SimCLE

Learning Sentence Embeddings via translation datasets

It is a implemention of pytorch vision

Train the sentence representations with your own translation dataset:
  The model file: model.py
  The trianing file: simcle_contrastive.py
  Setup instruction: ./main.sh
  
  The hyperparameter in main.sh: 
  
                    --save_model_path The path you want to save your pre-trained model. 
                    
                    --batch_size 
                    
                    --eval_step You evaluate your model every steps in training. 
                    
                    --queue_len The size of queueu containing negative samples, which is used to expand the comparaing objectives in contrastive learning. 
                    
                    --lr learning rate 
                    
                    --model_name The pre-training model name e.g. bert-base-uncased roberta-large
                    
                    --epochs 
                    
                    --task The pre-trainging task (default "plus" is represent the task combining contrastive objective and self-contrasting) 
                    
                    --zh_pmp The Chinese pre-training weight file （The pre-training model is based on the model "hfl/chinese-roberta-wwm-ext-large", and the "zh_pmp" file will update the weight of the based model.） 
                    
                    --eval_name The evaluating benchmark used in evaluation task. 
                    
                    --temp The temperature used in contrastive learning 
                    
                    --dropout 
                    
                    --zh_data_path The Chinese side path of parallel translation dataset. 
                    
                    --en_data_path The English side path of parallel translation dataset. 
                    
       
   
                    
                    
