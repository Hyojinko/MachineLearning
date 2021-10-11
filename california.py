import pandas as pd
import numpy as np
from matplotlib import cm

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

from sklearn import metrics
import warnings

warnings.filterwarnings(action='ignore')


def Missingvalue(df):
    # check missing value
    # only 'totla_bedrooms' has missing values, fill median
    df.total_bedrooms.fillna(df.total_bedrooms.median(), inplace=True)

    X = df.drop(['median_house_value'], axis=1)  # feature longitude, latitude, ..., ocean_proximity
    y = df.iloc[:, -2].copy()  # target median_house_value

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.to_list()  # numerical value
    cat_cols = X.select_dtypes(include=['object']).columns.to_list()  # categorical value

    return X, y, num_cols, cat_cols


def Combination_List(X, y, scale_feature, encode_feature):
    # 1. Scaler List : Standard, MinMax, maxAbs, Robust
    standard = StandardScaler()
    minMax = MinMaxScaler()
    maxAbs = MaxAbsScaler()
    robust = RobustScaler()
    scalers = {"standard scaler": standard, "minMax scaler": minMax, "maxAbs scaler": maxAbs, "robust scaler": robust}

    # 2. Encoder List : Label, One-hot
    label = LabelEncoder()
    oneHot = OneHotEncoder()
    encoders = {"label encoder": label, "one-hot encoder": oneHot}

    return X, y, scale_feature, encode_feature, scalers, encoders


def preprocessing(X, y, scale_feature, encode_feature, scalers, encoders):
    # combinations of scaler and encoder
    # scalers
    for scaler_key, scaler in scalers.items():

        X[scale_feature] = scaler.fit_transform(X[scale_feature])
        print("\n-----------------------------------------------------------------------")
        print(f'<   scaler: {scaler_key}    >')

        # encoders
        for encoder_key, encoder in encoders.items():
            # label encoder
            if encoder_key == "label encoder":
                X_1 = X.copy()

                def label_encoder(data):
                    for i in encode_feature:
                        data[i] = encoder.fit_transform(data[i])
                    return data

                X_label = label_encoder(X_1)

                cleaned_df = pd.concat([X_label, y], axis=1)
                print(f'\n      <   encoder: {encoder_key}   >')

                print(cleaned_df.head())

            # Onegit-hot encoder
            if encoder_key == "one-hot encoder":
                X_onehot = pd.get_dummies(X)

                cleaned_df = pd.concat([X_onehot, y], axis=1)
                print(f'\n      <   encoder: {encoder_key}   >')

                print(cleaned_df.head())

    return cleaned_df


# df=pd.read_csv("../Dataset/housing.csv")
df = pd.read_csv("housing.csv")

# preprocessing
X, y, scale_feature, encode_feature = Missingvalue(df)

X, y, scale_feature, encode_feature, scalers, encoders = Combination_List(X, y, scale_feature, encode_feature)

cleaned_df = preprocessing(X, y, scale_feature, encode_feature, scalers, encoders)

print(cleaned_df.info())

# Data
# Clustering
# DBSCAN

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score

house_location = cleaned_df[
    ['longitude', 'latitude', 'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND', 'ocean_proximity_ISLAND',
     'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']]
house_condition = cleaned_df[['housing_median_age', 'total_rooms', 'total_bedrooms']]
house_around = cleaned_df[['population', 'households', 'median_income']]

# function to calculate silhouette score
def silhouette_score(X, labels):
    sil_score = metrics.silhouette_score(X,labels,metric='euclidean')
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("For n_clusters =", n_clusters_, "The average silhouette score is :", sil_score)
    #compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, labels)

    fig,ax1 = plt.subplots()
    fig.set_size_inches(18,7)
    ax1.set_xlim([-0.1,1])
    ax1.set_ylim([0,len(X) + (n_clusters_+1)*10])
    y_lower = 10
    for i in range(n_clusters_):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters_)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),0,ith_cluster_silhouette_values,facecolor = color, edgecolor = color, alpha = 0.7 )
        ax1.text(-0.05, y_lower + 0.5*size_cluster_i,str(i))
        y_lower = y_upper + 10
    ax1.set_title("The silhouette plot for the various clusters")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=sil_score, color = "red", linestyle='--')
    ax1.set_yticks([])
    ax1.set_xticks([-0.1,0,0.2,0.4,0.6,0.8,1])

    plt.suptitle(("Silhouette analysis for clustering on sample data with n_clusters = %d" % n_clusters_),
                 fontsize=14, fontweight='bold')
    plt.show()
    return sil_score


def purity_score(target, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(target,y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0))/np.sum(contingency_matrix)

#variable for store silhouette score of dbscan
silhouette_dbscan = []
#variable for store purity score of dbscan
purity_dbscan = []
for i in range(1, 10, 1):
    dbscan = DBSCAN(eps=i * 0.1, min_samples=5)
    dbscan.fit(house_location)
    labels = dbscan.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % homogeneity_score(y, labels))
    print("Completeness: %0.3f" % completeness_score(y, labels))
    print("V-measure: %0.3f" % v_measure_score(y, labels))
    #calculate each silhouette score and store in silhouette score array
    sil_sco = silhouette_score(house_location, labels)
    silhouette_dbscan.append(sil_sco)
    #calculate each Purity score and store in purity score array
    purity_sco = purity_score(y, labels)
    purity_dbscan.append(purity_sco)
    print("Purity score: %.3f" % purity_sco)
    print("")


#select best silhouette score of dbscan
print("best silhouette score in DBSCAN clustering is %.3f" % np.max(silhouette_dbscan))

#select best purity score of dbscan
print("best purity score in DBSCAN clustering is %.3f" % np.max(purity_dbscan))
print("")


# MeanShift
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth

# To find optimize bandwith
best_bandwidth = estimate_bandwidth(house_location)
print('Best bandwidth :', round(best_bandwidth, 3))

meanshift = MeanShift(bandwidth=best_bandwidth)
cluster_labels = meanshift.fit_predict(house_location)
print('cluster labels :', np.unique(cluster_labels))
import matplotlib.pyplot as plt

house_location['meanshift_label'] = cluster_labels
centers = meanshift.cluster_centers_
unique_labels = np.unique(cluster_labels)

#Calculate silhouette score for meanshift clustering
silhouette_meanshift = silhouette_score(house_location, cluster_labels)
print("Silhouette score of MeanShift clustering is: %.3f"% silhouette_meanshift)

#Calculate purity score for meanshift clustering
purity_meanshift = purity_score(y, cluster_labels)
print("Purity score of MeanShift clustering is: %.3f"% purity_meanshift)








