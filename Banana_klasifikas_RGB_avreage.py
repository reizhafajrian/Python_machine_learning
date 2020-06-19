from glob import glob
import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix 
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
"""
NAMA Kelompok
Reizha fajiran   1810511078
M.Fadillah       1810511064
"""


"""Tugas akhir smester PCD Implementasi knn untuk 
menentukan kematangan pisang metodologi yang di gunakan
 yaitu ektrasi menggunakan rata rata nilai RGB dari citra"""
klasifikasis=[]#digunkan untuk menyimpan kalsifikasi data training
trainings=[]#menyimpan hasil extrasi data training
testing_data=[]#digunkan untuk menyimpan kalsifikasi data testing
testings=[]#menyimpan hasil extrasi data testing

def KNN():# fungsi ini di gunakan untuk melihat model apakah sudah berjalan sesuai dengan data dan menampilkan hasil dari pengolahan data
    knn=KNeighborsClassifier(n_neighbors=1) #define K=3
    knn.fit(data_average_training,klasifikasis)
    res = knn.predict(data_average_testing)
    results = confusion_matrix(testing_data, res) 
    response = np.concatenate([klasifikasis, res]).astype(np.float32)
    Dataset = np.concatenate([data_average_training, data_average_testing]).astype(np.float32)
    knn=KNeighborsClassifier(n_neighbors=1)
    score= cross_val_score(knn, Dataset, response, cv=2, scoring='accuracy')
    ax= plt.subplot()
    sns.heatmap(results, annot=True, ax = ax)
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['1 ','2']); ax.yaxis.set_ticklabels(['1', '2']);
    print(res)
    print("Cross Validation : ", score.mean())
    print ('Confusion Matrix :\n', results)
    print ('Accuracy Score :',accuracy_score(testing_data, res) )
    print ('Report : ')
    print (classification_report(testing_data, res))
    return res


def training(namafile):#fungsi untuk mengektrasi data training
        training=[]
        for filename in namafile:
            x=0
            if x<50:
                print(filename)
                myimg = cv.imread(filename)
                avg_color_per_row = np.average(myimg, axis=0)
                avg_color = np.average(avg_color_per_row, axis=0)
                data_warna=np.transpose(avg_color[0:3,np.newaxis])
                if namafile==data_sudah_matang or namafile==data_belum_matang:
                            if namafile==data_sudah_matang:
                                training.extend(data_warna)
                            else:
                                training.extend(data_warna)
          
            x+=1
        return training

def testing(namafile):#fungsi untuk mengektrasi data testing
      testing=[]
      for filename in namafile:
            x=0
            if x<50:
                print(filename)
                myimg = cv.imread(filename)
                avg_color_per_row = np.average(myimg, axis=0)
                avg_color = np.average(avg_color_per_row, axis=0)
                data_warna=np.transpose(avg_color[0:3,np.newaxis])
                if namafile==data_testing_belum_matang or namafile==data_testing_matang:
                            if namafile==data_testing_matang:
                                testing.extend(data_warna)
                            else:
                                testing.extend(data_warna)

            x+=1
      return testing

def klasifikasi_fungsi(namafile):#fungsi ini digunakan untuk menyimpan klasifikasi dari data training
    klasifikasi=[]
    for filename in namafile:
            if namafile==data_belum_matang:
                klasifikasi.append(1.0)
            else:
                klasifikasi.append(2.0)
    return klasifikasi

def prediksi_testing(namafile):#fungsi ini di gunakan untuk menyimpan data testing yang telah di ketahui untuk dibandingkan nantinta tingkat keakuratan data
    testing_predic=[]
    for filename in namafile:
            if namafile==data_testing_belum_matang:
                testing_predic.append(1.0)
            else:
                testing_predic.append(2.0)
    return testing_predic

def pemersatu_path(filename,filename1):#Fungsi ini di gunakan untuk menyatukan dua alamat/direktori testing
    tampilan_gambar=[]
    for i in filename:
        tampilan_gambar.append(i)
    for j in filename1:
        tampilan_gambar.append(j)
    return tampilan_gambar

def show_gambar(nama,new):#fungsi ini digunakan untuk menampilkan gambar sesuai prediksi
    pathfileindex=np.size(nama)
    res=new
    for i in range(0,pathfileindex+1):
        img = cv.imread("{}".format(nama[i-1]))
        plt.title(nama_pisang(res[i-1]))
        plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
        plt.show()
   
def nama_pisang(value):#fungsi ini untuk menentukan kematangan pisang berdasarkan res
    if value == 1.0:
        title="pisang Belum Matang"
    else:
        title="pisang Sudah Matang"
    return title
    
       

data_sudah_matang=sorted(glob('banana/matang/*.png'), key=lambda name: int(name[21:-4]))    #direktori training untuk pisang matang
data_belum_matang=sorted(glob('banana/mentah/*.jpg'), key=lambda name: int(name[21:-4]))#direktori training untuk pisang belum matang
data_testing_matang=sorted(glob('banana/test/test_matang/*.png'), key=lambda name: int(name[34:-4]))#direktori testing untuk pisang matang
data_testing_belum_matang=sorted(glob('banana/test/test_mentah/*.jpg'), key=lambda name: int(name[34:-4]))#direktori testing untuk pisang belum matang


trainings.extend(training(data_sudah_matang))#menambahkan extrasi dari data pisang matang fungsi ke dalam list
trainings.extend(training(data_belum_matang))#menambahkan extrasi dari data pisang belum matang dari fungsi ke dalam list

testings.extend(testing(data_testing_matang))#menambahakan data extrasi dari data testing pisang matang ke dalam list
testings.extend(testing(data_testing_belum_matang))#menambahakan data extrasi dari data testing pisang belum matang ke dalam list

klasifikasis.extend(klasifikasi_fungsi(data_sudah_matang))#mengkalsifikasikan objek berdasarkan angka dimana 2 matang dan 1 belum matang
klasifikasis.extend(klasifikasi_fungsi(data_belum_matang))

testing_data.extend(prediksi_testing(data_testing_matang))#mengklasifikasin data testing matang dan belum matang
testing_data.extend(prediksi_testing(data_testing_belum_matang))

data_average_testing=pd.DataFrame(testings).astype(np.float32)#mennyimpan hasil ektrasi kedalam data frame agar mudah untuk diolah
data_average_training=pd.DataFrame(trainings).astype(np.float32)

knn=KNeighborsClassifier(n_neighbors=1) #define K=3
knn.fit(data_average_training,klasifikasis)
res = knn.predict(data_average_testing)
results = confusion_matrix(testing_data, res) 
response = np.concatenate([klasifikasis, res]).astype(np.float32)
Dataset = np.concatenate([data_average_training, data_average_testing]).astype(np.float32)
knn=KNeighborsClassifier(n_neighbors=1)
score= cross_val_score(knn, Dataset, response, cv=2, scoring='accuracy')
ax= plt.subplot()
sns.heatmap(results, annot=True, ax = ax)
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['1 ','2']); ax.yaxis.set_ticklabels(['1', '2']);
print(res)
print("Cross Validation : ", score.mean())
print ('Confusion Matrix :\n', results)
print ('Accuracy Score :',accuracy_score(testing_data, res) )
print ('Report : ')
print (classification_report(testing_data, res))


new=KNN()
pemersatu_alamat=pemersatu_path(data_testing_matang, data_testing_belum_matang)

show_gambar(pemersatu_alamat,new)



