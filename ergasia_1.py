import numpy as np  # Εισαγωγή της βιβλιοθήκης numpy για αριθμητικούς υπολογισμούς
import seaborn as sns  # Εισαγωγή της βιβλιοθήκης seaborn για οπτικοποίηση δεδομένων
import pandas as pd  # Εισαγωγή της βιβλιοθήκης pandas για διαχείριση δεδομένων
import matplotlib.pyplot as plt  # Εισαγωγή της βιβλιοθήκης matplotlib για δημιουργία γραφικών παραστάσεων
from sklearn import tree  # Εισαγωγή της βιβλιοθήκης scikit-learn για αλγόριθμους μηχανικής μάθησης
from sklearn.model_selection import train_test_split, cross_val_score, KFold  # Εισαγωγή συναρτήσεων για διαχωρισμό και επικύρωση δεδομένων
from sklearn.tree import DecisionTreeClassifier  # Εισαγωγή του αλγορίθμου Δέντρου Απόφασης
from sklearn.neighbors import KNeighborsClassifier  # Εισαγωγή του αλγορίθμου k-Nearest Neighbors (k-NN)
from sklearn.preprocessing import StandardScaler  # Εισαγωγή του StandardScaler για κανονικοποίηση δεδομένων
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score  # Εισαγωγή μετρικών για αξιολόγηση του μοντέλου

# Εισαγωγή και εμφάνιση των δεδομένων
phishing_df = pd.read_csv('./phishing+websites/training_dataset.csv')  # Ανάγνωση του αρχείου CSV που περιέχει τα δεδομένα
phishing_df.dropna(inplace=True)  # Αφαίρεση γραμμών που περιέχουν κενά δεδομένα
print(phishing_df.head())  # Εκτύπωση των πρώτων 5 γραμμών του dataframe για να δούμε τα δεδομένα

# Μετατροπή κατηγορικών δεδομένων σε αριθμητικά (αν υπάρχει ανάγκη)
phishing_data = pd.get_dummies(phishing_df, dtype=int)  # Μετατροπή κατηγορικών δεδομένων σε δυαδικές (one-hot) μεταβλητές
print(phishing_data.head())  # Εκτύπωση των πρώτων 5 γραμμών μετά την μετατροπή

# Κανονικοποίηση δεδομένων
# Κάνει όλα τα δεδομένα να έχουν την ίδια κλίμακα και το μοντέλο μας να μην επηρεάζεται από την κλίμακα των χαρακτηριστικών
feature_names = phishing_data.drop(columns=['Result']).columns  # Αποθήκευση των ονομάτων των χαρακτηριστικών (χωρίς τον στόχο 'Result')
scaler = StandardScaler()  # Δημιουργία αντικειμένου για κανονικοποίηση
scaled_features = scaler.fit_transform(phishing_data.drop(columns=['Result']))  # Κανονικοποίηση των χαρακτηριστικών
scaled_phishing_data = pd.DataFrame(scaled_features, columns=feature_names)  # Δημιουργία dataframe με τα κανονικοποιημένα χαρακτηριστικά
scaled_phishing_data['Result'] = phishing_data['Result']  # Προσθήκη της στήλης στόχου ('Result') στο κανονικοποιημένο σύνολο δεδομένων

# Διαχωρισμός χαρακτηριστικών και στόχου
X = scaled_phishing_data.drop(columns=['Result']).values  # Ορισμός των χαρακτηριστικών (X) χωρίς την στήλη στόχου
y = scaled_phishing_data['Result'].values  # Ορισμός της στήλης στόχου (y)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # Διαχωρισμός σε σύνολο εκπαίδευσης και δοκιμής (70%-30%)
 
# Απόφαση Δέντρου Απόφασης
dt = DecisionTreeClassifier(max_leaf_nodes=30, random_state=1)  # Δημιουργία και παραμετροποίηση του μοντέλου Δέντρου Απόφασης
dt.fit(x_train, y_train)  # Εκπαίδευση του μοντέλου με τα δεδομένα εκπαίδευσης
dt_predictions = dt.predict(x_test)  # Πρόβλεψη με το εκπαιδευμένο μοντέλο για τα δεδομένα δοκιμής

# Οπτικοποίηση Δέντρου Απόφασης
plt.figure(figsize=(13,8))  # Ορισμός μεγέθους γραφήματος
tree_vis = tree.plot_tree(dt, filled=True)  # Οπτικοποίηση του Δέντρου Απόφασης
plt.show()  # Εμφάνιση του γραφήματος

# Αξιολόγηση Δέντρου Απόφασης
dt_accuracy = accuracy_score(y_test, dt_predictions)  # Υπολογισμός της ακρίβειας του μοντέλου
dt_recall = recall_score(y_test, dt_predictions, pos_label=1)  # Υπολογισμός του δείκτη recall
dt_f1 = f1_score(y_test, dt_predictions, pos_label=1)  # Υπολογισμός του δείκτη F1
print(f"Decision Tree - Accuracy: {dt_accuracy}, Recall: {dt_recall}, F1-Score: {dt_f1}")  # Εκτύπωση των αποτελεσμάτων αξιολόγησης

# Εφαρμογή k-NN
knn = KNeighborsClassifier(n_neighbors=5)  # Δημιουργία του μοντέλου k-NN με 5 γείτονες
knn.fit(x_train, y_train)  # Εκπαίδευση του μοντέλου k-NN με τα δεδομένα εκπαίδευσης
knn_predictions = knn.predict(x_test)  # Πρόβλεψη με το μοντέλο k-NN για τα δεδομένα δοκιμής

# Αξιολόγηση k-NN
knn_accuracy = accuracy_score(y_test, knn_predictions)  # Υπολογισμός της ακρίβειας του μοντέλου k-NN
knn_recall = recall_score(y_test, knn_predictions, pos_label=1)  # Υπολογισμός του δείκτη recall για το k-NN
knn_f1 = f1_score(y_test, knn_predictions, pos_label=1)  # Υπολογισμός του δείκτη F1 για το k-NN
print(f"k-NN - Accuracy: {knn_accuracy}, Recall: {knn_recall}, F1-Score: {knn_f1}")  # Εκτύπωση των αποτελεσμάτων αξιολόγησης για το k-NN

# Πίνακες Σύγχυσης
dt_cm = confusion_matrix(y_test, dt_predictions)  # Δημιουργία πίνακα σύγχυσης για το Δέντρο Απόφασης
knn_cm = confusion_matrix(y_test, knn_predictions)  # Δημιουργία πίνακα σύγχυσης για το k-NN

# Εμφάνιση των πινάκων σύγχυσης με γραφική απεικόνιση
plt.figure(figsize=(12, 5))  # Ορισμός μεγέθους γραφήματος
plt.subplot(1, 2, 1)  # Δημιουργία του πρώτου υπογραφήματος για το Δέντρο Απόφασης
sns.heatmap(dt_cm, annot=True, cmap="Blues", fmt="d", xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])  # Οπτικοποίηση πίνακα σύγχυσης για το Δέντρο Απόφασης
plt.title("Decision Tree Confusion Matrix")  # Τίτλος του γραφήματος
plt.xlabel("Predicted")  # Ετικέτα άξονα Χ
plt.ylabel("Actual")  # Ετικέτα άξονα Υ

plt.subplot(1, 2, 2)  # Δημιουργία του δεύτερου υπογραφήματος για το k-NN
sns.heatmap(knn_cm, annot=True, cmap="Blues", fmt="d", xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])  # Οπτικοποίηση πίνακα σύγχυσης για το k-NN
plt.title("k-NN Confusion Matrix")  # Τίτλος του γραφήματος
plt.xlabel("Predicted")  # Ετικέτα άξονα Χ
plt.ylabel("Actual")  # Ετικέτα άξονα Υ
plt.tight_layout()  # Ρύθμιση του layout για καλύτερη εμφάνιση
plt.show()  # Εμφάνιση του γραφήματος

# Διασταυρούμενη Επικύρωση (Cross-validation)
dt_cv_scores = cross_val_score(dt, X, y, cv=10)  # Εκτέλεση 10-fold cross-validation για το Δέντρο Απόφασης
knn_cv_scores = cross_val_score(knn, X, y, cv=10)  # Εκτέλεση 10-fold cross-validation για το k-NN

# Εκτύπωση των αποτελεσμάτων διασταυρούμενης επικύρωσης
print(f"Decision Tree Cross-Validation Accuracy: {np.mean(dt_cv_scores):.2f}")  # Μέση ακρίβεια για το Δέντρο Απόφασης
print(f"k-NN Cross-Validation Accuracy: {np.mean(knn_cv_scores):.2f}")  # Μέση ακρίβεια για το k-NN

# Δοκιμή διαφόρων τιμών για το k στον k-NN
k_values = range(1, 31)  # Εύρος τιμών για το k από 1 έως 30
mean_accuracies = []  # Λίστα για αποθήκευση των μέσων ακριβειών για κάθε τιμή του k
trained_models = {}  # Λεξικό για αποθήκευση των εκπαιδευμένων μοντέλων

# Εκπαίδευση και αξιολόγηση του k-NN για κάθε τιμή του k
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)  # Δημιουργία του μοντέλου k-NN για την τρέχουσα τιμή του k
    kf = KFold(n_splits=5, shuffle=True, random_state=1)  # Δημιουργία αντικειμένου για 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=kf)  # Εκτέλεση cross-validation

    mean_accuracy = np.mean(cv_scores)  # Υπολογισμός της μέσης ακρίβειας
    mean_accuracies.append(mean_accuracy)  # Αποθήκευση της μέσης ακρίβειας
    trained_models[k] = model.fit(X, y)  # Εκπαίδευση του μοντέλου σε όλο το σύνολο δεδομένων

# Εκτύπωση μέσου ακριβείας για κάθε k
print("Mean Accuracies for each k:", mean_accuracies)

# Γραφική παράσταση ακρίβειας ανά τιμή του k
plt.figure(figsize=(10, 6))  # Ορισμός μεγέθους γραφήματος
plt.plot(k_values, mean_accuracies, marker='o')  # Σχεδίαση γραμμής για την ακρίβεια ανά τιμή του k
plt.title("Mean Accuracy for k-NN (k = 1 to 30)")  # Τίτλος γραφήματος
plt.xlabel("k (Number of Neighbors)")  # Ετικέτα άξονα Χ
plt.ylabel("Mean Accuracy")  # Ετικέτα άξονα Υ
plt.grid()  # Ενεργοποίηση του πλέγματος
plt.show()  # Εμφάνιση του γραφήματος

# Δημιουργία πινάκων σύγχυσης για συγκεκριμένες τιμές του k
k_values_confusion = [3, 5, 15, 25]  # Παράδειγμα τιμών k για αξιολόγηση

# Επανάληψη για κάθε επιλεγμένη τιμή του k
for k in k_values_confusion:
    model = trained_models[k]  # Ανάκτηση του εκπαιδευμένου μοντέλου για την τιμή του k
    y_pred = model.predict(X)  # Πρόβλεψη για τα δεδομένα
    cm = confusion_matrix(y, y_pred)  # Δημιουργία πίνακα σύγχυσης

    class_labels = ["Legitimate", "Phishing"]  # Ετικέτες για τις κλάσεις

    # Γραφική απεικόνιση πίνακα σύγχυσης
    plt.figure(figsize=(8, 6))  # Ορισμός μεγέθους γραφήματος
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",  # Οπτικοποίηση πίνακα σύγχυσης
                xticklabels=class_labels, yticklabels=class_labels,
                linewidths=1, linecolor='black', annot_kws={"size": 12})
    plt.xlabel("Predicted", fontsize=14, fontweight='bold')  # Ετικέτα άξονα Χ
    plt.ylabel("Actual", fontsize=14, fontweight='bold')  # Ετικέτα άξονα Υ
    plt.title(f"Confusion Matrix for {k}-NN Classifier")  # Τίτλος γραφήματος
    plt.show()  # Εμφάνιση του γραφήματος
