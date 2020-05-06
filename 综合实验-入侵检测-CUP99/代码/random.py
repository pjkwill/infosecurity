# 数据降维的时候，可以选择将数据分类型进行降维，也可以全部的降维，采用方法有ICA和PCA代码
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler  # install scipy package
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import ensemble
from sklearn.decomposition import PCA as sklearnPCA


class IntrusionDetector:

    def __init__(self, train_data_path, test_kdd_path):
        self.train_kdd_path = train_data_path
        self.test_kdd_path = test_kdd_path

        self.train_kdd_data = []
        self.test_kdd_data = []

        self.train_kdd_numeric = []
        self.test_kdd_numeric = []

        self.train_kdd_binary = []
        self.test_kdd_binary = []

        self.train_kdd_nominal = []
        self.test_kdd_nominal = []

        self.train_kdd_label_2classes = []
        self.test_kdd_label_2classes = []
        #read data from file
        self.get_data()


    def get_data(self):
        col_names = ["duration","protocol_type","service","flag","src_bytes",
            "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
            "logged_in","num_compromised","root_shell","su_attempted","num_root",
            "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
            "is_host_login","is_guest_login","count","srv_count","serror_rate",
            "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
            "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
            "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
            "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
        self.train_kdd_data = pd.read_csv(self.train_kdd_path, header=None, names = col_names)
        self.test_kdd_data = pd.read_csv(self.test_kdd_path, header=None, names = col_names)
        self.train_kdd_data.describe()

    # To reduce labels into "Normal" and "Abnormal"
    def get_2classes_labels(self):
        label_2class = self.train_kdd_data['label'].copy()
        self.train_kdd_label_2classes = label_2class.values.reshape((label_2class.shape[0], 1))

        label_2class = self.test_kdd_data['label'].copy()
        self.test_kdd_label_2classes = label_2class.values.reshape((label_2class.shape[0], 1))

    def preprocessor(self):
        # prepare 2 classes label for "abnormal" and "normal"
        self.get_2classes_labels()

        nominal_features = ["protocol_type", "service", "flag"]  # [1, 2, 3]
        binary_features = ["land", "logged_in", "root_shell", "su_attempted", "is_host_login", "is_guest_login",]  # [6, 11, 13, 14, 20, 21]
        numeric_features = [
            "duration", "src_bytes",
            "dst_bytes", "wrong_fragment", "urgent", "hot",
            "num_failed_logins", "num_compromised", "num_root",
            "num_file_creations", "num_shells", "num_access_files",
            "num_outbound_cmds", "count", "srv_count", "serror_rate",
            "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
            "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
        ]

        #convert nominal features to numeric features
        #nominal features: ["protocol_type", "service", "flag"]
        self.train_kdd_nominal = self.train_kdd_data[nominal_features].astype(float)
        self.test_kdd_nominal = self.test_kdd_data[nominal_features].astype(float)
        # normalize
        # self.train_kdd_nominal = StandardScaler().fit_transform(self.train_kdd_nominal)
        # self.test_kdd_nominal = StandardScaler().fit_transform(self.test_kdd_nominal)

        self.train_kdd_binary = self.train_kdd_data[binary_features].astype(float)
        self.test_kdd_binary = self.test_kdd_data[binary_features].astype(float)
        # normalize
        # self.train_kdd_binary = StandardScaler().fit_transform(self.train_kdd_binary)
        # self.test_kdd_binary = StandardScaler().fit_transform(self.test_kdd_binary)

        # Standardizing and scaling numeric features
        self.train_kdd_numeric = self.train_kdd_data[numeric_features].astype(float)
        self.test_kdd_numeric = self.test_kdd_data[numeric_features].astype(float)
        # normalize
        self.train_kdd_numeric = StandardScaler().fit_transform(self.train_kdd_numeric)
        self.test_kdd_numeric = StandardScaler().fit_transform(self.test_kdd_numeric)

    def format_data(self):

        kdd_train_data = np.concatenate([self.train_kdd_numeric, self.train_kdd_binary, self.train_kdd_nominal], axis=1)
        kdd_test_data = np.concatenate([self.test_kdd_numeric, self.test_kdd_binary, self.test_kdd_nominal], axis=1)

        kdd_train_data = np.concatenate([kdd_train_data, self.train_kdd_label_2classes],axis=1)
        # kdd_test_data = np.concatenate([self.test_kdd_numeric, self.test_kdd_binary, self.test_kdd_nominal, self.test_kdd_label_2classes], axis=1)
        kdd_test_data = np.concatenate([kdd_test_data, self.test_kdd_label_2classes], axis=1)
        self.X_train, self.X_test, y_train, y_test = kdd_train_data[:, :-1], kdd_test_data[:, :-1], kdd_train_data[:,-1], kdd_test_data[:, -1]

        data_pca = sklearnPCA(n_components=15)
        data_pca = data_pca.fit(self.X_train)
        # numeric_pca = numeric_pca.fit(np.concatenate((self.train_kdd_numeric, self.test_kdd_numeric), axis=0))
        self.X_train = data_pca.transform(self.X_train)
        self.X_test = data_pca.transform(self.X_test)

        self.y_train = np.array(list(map(int, y_train)))
        self.y_test = np.array(list(map(np.int64, y_test)))

    def predicting(self, model, model_name):
        # Predict
        predicts = model.predict(self.X_test)
        print("Classifier:")
        accuracy = accuracy_score(self.y_test, predicts)
        print("Accuracy: ", accuracy)

        con_matrix = confusion_matrix(self.y_test, predicts, labels=[0, 1])
        # con_matrix = confusion_matrix(y_test, predicts, labels=["normal.", "abnormal."])
        print("confusion matrix:")
        print(con_matrix)
        precision = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[1][0])
        recall = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
        tpr = recall
        fpr = con_matrix[1][0] / (con_matrix[1][0] + con_matrix[1][1])
        print("Precision:", precision)
        print("Recall:", recall)
        print("TPR:", tpr)
        print("FPR:", fpr)


    def random_forest_classifier(self):
        # model = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=35, criterion="entropy")
        model = ensemble.RandomForestClassifier()
        model.fit(self.X_train, self.y_train)
        self.predicting(model, "RFC")

def main():
    # Data path
    #cwd = os.getcwd()  # current directory path
    kdd_data_path_train ="kddcup.data_10_percent_corrected_new.csv"
    kdd_data_path_test ="kddcup.data.corrected.csv"
    i_detector = IntrusionDetector(kdd_data_path_train, kdd_data_path_test)
    i_detector.preprocessor()
    i_detector.format_data()
    i_detector.random_forest_classifier()


if __name__ == '__main__':
    main()