import matplotlib.pyplot as plt     # For curve plotting
import numpy as np                  # Data processing, linear algebra
import pandas as pd                 # CSV file input


def train_test_split(X, Y, test_size):
    m=X.shape[0]
    positions=np.random.permutation(m)
    num=int(test_size*m)
    X_test=X[positions[0:num]]
    Y_test=Y[positions[0:num]]
    X_train=X[positions[num:]]
    Y_train=Y[positions[num:]]
    return X_train, X_test, Y_train, Y_test


def sigmoid(x):
    x=np.maximum(-40, x)    # To avoid overflow
    x=1+np.exp(-x)
    x=np.reciprocal(x)
    return x


# Transform the matrix to do convolution as a normal dot product
def transform(x):
    s=int(round((np.sqrt(x.shape[0]))/2))
    x=np.reshape(x, (2*s,2*s))
    num=s*s
    ans=np.zeros((9,num))
    
    x=np.concatenate((x, np.zeros((2*s,1))), axis=1)
    x=np.concatenate((x, np.zeros((1,2*s+1))), axis=0)
    x[2*s,:]=x[2*s-1,:]
    x[:,2*s]=x[:,2*s-1]
    
    for j in range(0,num):
        m=2*(j//s)
        n=2*(j%s)
        ans[:,j]=x[m:m+3,n:n+3].flatten()
    return ans


# To invert the transform function that was applied
def inverse(x):
    s=x.shape[1]
    ans=np.zeros((11,11))
    for j in range(0,25):
        m=2*(j//5)
        n=2*(j%5)
        ans[m:m+3,n:n+3]+=x[:,j].reshape((3,3))
    return ans[0:10,0:10].flatten()


def image_plotter(img_arr, actual, predicted):
    actual=(1+actual)%10
    predicted=(1+predicted)%10
    img_matrix=np.zeros((20,20))
    for i in range(0,20):
        img_matrix[:,i]=img_arr[20*i:20*(i+1)]
    plt.imshow(img_matrix)
    plt.title('Actual is %d and predicted is %d' %(actual, predicted))
    plt.show()


def calc_error(U, V, W, X_dash, Y, print_faulty):
    ans=0
    num=np.zeros(10, dtype=int)
    HA=np.ones((num_of_filters,9,25))
    HB=np.ones(num_of_filters*25+1)
    for pos in range(0,X_dash.shape[0]):
        X=transform(X_dash[pos,:])
        ha=sigmoid(np.dot(U,X))
        for i in range(0,num_of_filters):
            HA[i]=transform(ha[i])
            
        hb=sigmoid(np.dot(V,np.sum(HA,axis=0)))
        HB[1:]=hb.flatten()
        O_dash=np.dot(W,HB)
        sum=np.sum(np.exp(O_dash))
        O=np.exp(O_dash)/sum
        p1, p2 = np.argmax(O), np.argmax(Y[pos,:])
        if p1 != p2:
            ans+=1
            if print_faulty == True:
                if num[p2] < 2:
                    image_plotter(X_dash[pos,:], p2, p1)
                    num[p2]+=1
                    
    return ans*100/X_dash.shape[0]


def gradient(x):
    return np.multiply(x, 1-x)


# Reading image data and label from original files
orig_data=np.array(pd.read_csv('data.txt', header=None))
orig_label=np.array(pd.read_csv('label.txt', header=None))


test_frac=0.2           # Fraction of test data from overall data
val_frac=0.2            # Fraction of validation data from remaining data after removal of test data
num_of_filters=50       # Number of filters
batch_size=25           # Batch size
alpha=0.01              # Learning rate
num_of_epochs=100       # Number of epochs
num_of_trials=5         # Number of trials


tra_error=np.zeros((num_of_trials,num_of_epochs))
val_error=np.zeros((num_of_trials,num_of_epochs))

for trial in range(0,num_of_trials):
    print("\n\nPerforming trial %d of %d trials" %(trial+1, num_of_trials))
    rem_data, test_data, rem_label, test_label = train_test_split(orig_data, orig_label, test_size=test_frac)
    tra_data, val_data, tra_label, val_label = train_test_split(rem_data, rem_label, test_size=val_frac)
    HA=np.ones((num_of_filters,9,25))
    HB=np.ones(num_of_filters*25+1)
    U=np.random.normal(0,0.1,(num_of_filters,9))
    V=np.random.normal(0,0.1,(num_of_filters,9))
    W=np.random.normal(0,0.1,(10,num_of_filters*25+1))
    del_ha=np.zeros((num_of_filters,100))
    del_HA=np.zeros((num_of_filters,9,25))
    del_HB=np.zeros(num_of_filters*25+1)
    del_U=np.zeros((num_of_filters,9))
    del_V=np.zeros((num_of_filters,9))
    del_W=np.zeros((10,num_of_filters*25+1))

    for epoch in range(0,num_of_epochs):
        print("Performing epoch %d of %d epochs\r" %(epoch+1, num_of_epochs), end='\r')
        start=0
        end=min(start+batch_size,tra_data.shape[0])
        while start<tra_data.shape[0]:
            del_U.fill(0)
            del_V.fill(0)
            del_W.fill(0)
            for pos in range(start,end):
                X=transform(tra_data[pos,:])
                ha=sigmoid(np.dot(U,X))
                for i in range(0,num_of_filters):
                    HA[i]=transform(ha[i])
                
                sum_HA=np.sum(HA,axis=0)
                hb=sigmoid(np.dot(V,sum_HA))
                HB[1:]=hb.flatten()
                O_dash=np.dot(W,HB)
                sum=np.sum(np.exp(O_dash))
                O=np.exp(O_dash)/sum
                    
                del_W+=np.outer((O-tra_label[pos,:]),HB)
                
                del_HB=np.dot(W.T,O-tra_label[pos,:])
                del_hb=np.reshape(del_HB[1:],(num_of_filters,25))
                
                derivative_hb=gradient(hb)
                del_V+=np.dot(np.multiply(del_hb, derivative_hb), sum_HA.T)
                
                del_HA[:]=np.dot(V.T, np.multiply(del_hb, derivative_hb))
                del_ha[:]=inverse(del_HA[0])
                
                del_U+=np.dot(np.multiply(del_ha, gradient(ha)), X.T)

            del_U/=(end-start)
            del_V/=(end-start)
            del_W/=(end-start)
            U=U-alpha*del_U
            V=V-alpha*del_V
            W=W-alpha*del_W
            start=end
            end=min(start+batch_size,tra_data.shape[0])
            
        tra_error[trial][epoch]=(calc_error(U, V, W, tra_data, tra_label, False))
        val_error[trial][epoch]=(calc_error(U, V, W, val_data, val_label, False))
        
test_error=(calc_error(U, V, W, test_data, test_label, True))


avg_tra_error=np.mean(tra_error,axis=0)
avg_val_error=np.mean(val_error,axis=0)
std_tra_error=np.std(tra_error,axis=0)
std_val_error=np.std(val_error,axis=0)

plt.errorbar(range(1,num_of_epochs+1),avg_tra_error, yerr=std_tra_error)
plt.title('Mean training error over %d trials' %num_of_trials)
plt.legend(('training',))
plt.xlabel('number of epochs')
plt.ylabel('percentage error')
plt.show()

plt.errorbar(range(1,num_of_epochs+1),avg_val_error, yerr=std_val_error, color='#ff7f0e')
plt.title('Mean validation error over %d trials' %num_of_trials)
plt.legend(('validation',))
plt.xlabel('number of epochs')
plt.ylabel('percentage error')
plt.show()

plt.errorbar(range(1,num_of_epochs+1),avg_tra_error, yerr=std_tra_error)
plt.errorbar(range(1,num_of_epochs+1),avg_val_error, yerr=std_val_error)
plt.title('Mean training and validation error over %d trials' %num_of_trials)
plt.legend(('training','validation'))
plt.xlabel('number of epochs')
plt.ylabel('percentage error')
plt.show()

print('For %d epochs over %d trials:' %(num_of_epochs, num_of_trials))
print('Mean training error is %f percent' %(avg_tra_error[num_of_epochs-1]))
print('Mean validation error is %f percent' %(avg_val_error[num_of_epochs-1]))