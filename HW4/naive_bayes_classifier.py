import tarfile
  
# open file
file = tarfile.open('./languageID.tgz')
  
# extracting file
file.extractall('./train_data')
  
file.close()
y_label = ['e', 'j', 's']
import numpy as np
char_count = np.zeros([3,20,27])
for k, label in enumerate(y_label):
    print(label);
    #print(k);
    #char_count = np.zeros([3,10,27])
    for i in range(20):
        print('train_data/languageID/%s'%label+'%d'%i+'.txt')
        f = open('train_data/languageID/%s' % label+ '%d' % i+'.txt')
        data = f.read();
        for j in data:
            if((j!='\n')):
                if((j == ' ')):
                    char_count[k][i][26] += 1;
                else:
                    char_count[k][i][ord(j)-97] += 1;
        #print("space_count =",char_count[26]);
        print("count =", char_count[k][i])

likelihood = np.zeros([3,27]);
alpha = 0.5;
for i in range(3):
    print("char count sum :", np.sum(char_count[i][:10],axis=0))
    likelihood[i]= (np.sum(char_count[i][:10],axis=0) + alpha)/(np.sum(np.sum(char_count[i][:10],axis=0)) + 27*alpha) 

def predict(x):
    prob = np.zeros(3);
    for i in range(3):
        for j in range(27):
            prob[i] = prob[i] + x[j]*np.log(likelihood[i][j])
    prob_label = np.log(1/3);
    print(prob)
    print(prob+prob_label); 
    return np.argmax(prob+prob_label);
    
predict(char_count[0][10])

#test data
pred = np.zeros([3, 10])
for i in range(3):
    for j in range(10):
        pred[i][j] = predict(char_count[i][j+10]);

print(pred)