import PIL.Image
import cv2
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
torch.set_default_tensor_type("torch.cuda.FloatTensor")

layer_dims = [1600, 50, 2]

font=cv2.FONT_HERSHEY_SIMPLEX

def non_linear_activation(Z, activation = "ReLU"):
    if activation == "tanh" :
        Z = torch.tanh(Z)
        
    if activation == "ReLU" :
        zero = torch.cuda.FloatTensor([0])
        Z = torch.max(zero.expand_as(Z), Z)
        
    if activation == "softmax":
        Z = torch.exp(Z)
        Z_sum = torch.sum(Z, dim=0)
        Z = Z/Z_sum
    
    return Z

def forward_propagation(X, parameters):
  
    L = len(layer_dims)
    cache = {}
    cache["A0"] = X
    A = X
    
    for l in range(1, L):
        cache["Z" + str(l)] = torch.mm(parameters["W" + str(l)], A) + parameters["b" + str(l)]
        cache["A" + str(l)] = non_linear_activation(cache["Z" + str(l)], activation = "tanh")

        assert(cache["Z" + str(l)].shape == cache["A" + str(l)].shape)
        A = cache["A" + str(l)]
        
    cache["A" + str(L - 1)] = non_linear_activation(cache["Z" + str(L - 1)], activation = "softmax")
    assert(cache["Z" + str(L - 1)].shape == cache["A" + str(L - 1)].shape)

    return cache


def detect_face(test_X, layer_dims):
    
    while True:
        indx = int(input("Enter the numeber < 2000 : "))
        if indx == -1:
          break
        X = test_X[:, indx].view(1600, 1)
        print(X.shape)
        img = X.contiguous().cpu().numpy().reshape(40, 40)
        cache = forward_propagation(X, parameters)
        out = list(cache["A2"].cpu().numpy().squeeze())
        maximum = out.index(max(out))

        plt.title("{}".format(classes[str(maximum)]))
        plt.imshow(img, cmap="gray")
        plt.show()

        print("Probability distribution : ", out)

def real_time_detection():
  
    parameters = pickle.load(open("trained_parameters_65_69.pickle", "rb"))
    classes = {"0":"Female", "1":"Male"}
    
    detector = cv2.CascadeClassifier('E:\ML project\haarcascade_frontalface_default.xml')
    cam=cv2.VideoCapture(0)

    while(True):
        ret,img=cam.read();
        grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=detector.detectMultiScale(grey);
        for(x,y,w,b) in faces:
            cv2.rectangle(img,(x-20,y-40),(x+w+20,y+b+20),(255,0,0),2)
            new = cv2.resize(grey[y:y+b,x:x+w], (40, 40)) / 255
            X = torch.from_numpy(np.reshape(new, (1600, 1))).type(torch.cuda.FloatTensor)
            cache = forward_propagation(X, parameters)
            out = list(cache["A2"].cpu().numpy().squeeze())
            maximum = out.index(max(out))
            out1=float("{0:.1f}".format(max(out)*100))
            cv2.putText(img,(classes[str(maximum)])+"  "+str(out1)+"%",(x,y+b+45),font,1,(0,255,0),2)
            
        cv2.imshow("Camera",img);
        if(cv2.waitKey(1)==ord('q')):
            break;
    cam.release()
    cv2.destroyAllWindows()    
    exit(0)


real_time_detection()


