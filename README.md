🛡️ Phishing Website Detection

Αυτό το project χρησιμοποιεί μηχανική μάθηση για την ανίχνευση phishing ιστοσελίδων βασισμένο σε χαρακτηριστικά των URLs. Περιλαμβάνει ανάλυση, επεξεργασία δεδομένων, εκπαίδευση μοντέλων και αξιολόγηση με χρήση δύο βασικών αλγορίθμων:

    🌳 Decision Tree Classifier

    🤝 k-Nearest Neighbors (k-NN)
Προκαταρκτική επεξεργασία

    Ανάγνωση και καθαρισμός δεδομένων

    One-hot encoding

    Κανονικοποίηση με StandardScaler

Εκπαίδευση μοντέλων

    Decision Tree (max_leaf_nodes=30)

    k-NN με k=5 και βελτιστοποίηση για k=1...30

Αξιολόγηση

    Accuracy, Recall, F1-Score

    Confusion Matrix

    10-fold & 5-fold Cross-validation

Οπτικοποίηση

    Heatmaps για confusion matrices

    Γράφημα βελτιστοποίησης k στον k-NN

    Γραφική αναπαράσταση δέντρου απόφασης
