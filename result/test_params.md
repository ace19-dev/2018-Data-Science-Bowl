| No | label size | image size | batch size | conv padding | epoch  | use 64 channel |  activation func. | submit result |
|----|------|------------|------------|---------------|--------|------------------|------------------|---------------|
| 1  |    | 256*256    | 32         | 10            | 3, 10  | elu             |                     |  0.306         |
| 2  |    | -          | 16         | 10            | 15, 50 | elu              |   |0.346         |
| 3  |    | 384*384    | 16         | 10            | 50, 50 | elu              |    |0.336         |
  
    
      
          
             
             

unet_sample_352.py 

| No | data | image size | batch size | learning step | epoch  | activation func. | submit result |
|----|------|------------|------------|---------------|--------|------------------|---------------|
| 1  | x3   | 256*256    | 32         | 10            | 3, 10  | elu              | 0.306         |
| 2  | x3   | -          | 16         | 10            | 15, 50 | elu              | 0.346         |
| 3  | x3   | 384*384    | 16         | 10            | 50, 50 | elu              | 0.336         |

- activation func.  
동일한 조건에서 결과는 relu < leaky relu < elu 순.
