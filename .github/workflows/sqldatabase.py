import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    username="root",
    password="pass",
    database="networkdatabase"
)

mycursor = db.cursor(buffered=True)

Q1 = "CREATE TABLE Models (ID int PRIMARY KEY NOT NULL AUTO_INCREMENT, modelName VARCHAR(255) NOT NULL, numHidden int NOT NULL, epochs int NOT NULL, mini_batch_size int NOT NULL, eta float NOT NULL)"
Q2 = "CREATE TABLE Accuracy (modelID int PRIMARY KEY, FOREIGN KEY(modelID) REFERENCES Models(ID), currentAccuracy float NOT NULL, maxAccuracy float NOT NULL, maxAccuracyEpoch int NOT NULL, accuracyArray VARCHAR(4095) NOT NULL)"

Q3 = "ALTER TABLE Models ADD CONSTRAINT CHK_Models CHECK(numHidden>=0 AND epochs>=1 AND mini_batch_size >=1 AND eta>=0)"
Q4 = "ALTER TABLE Accuracy ADD CONSTRAINT CHK_Accuracy CHECK(currentAccuracy>=0 AND currentAccuracy<=100 AND maxAccuracy>=0 AND maxAccuracy<=100 AND maxAccuracyEpoch>=0)"


# mycursor.execute(Q3)
# mycursor.execute(Q4)
# mycursor.execute("DESCRIBE Models")
# mycursor.execute("DESCRIBE Accuracy")

insertModel = "INSERT INTO Models (modelName, numHidden, epochs, mini_batch_size, eta) VALUES (%s,%d,%d,%d,%f)"

#mycursor.execute(insertModel,networkName, str(self.sizes[1]), str(self.epochs), str(self.mini_batch_size), str(self.eta))
#mycursor.execute("ALTER TABLE Accuracy CHANGE accuracyArray accuracyArray  VARCHAR(4095) NOT NULL")

for x in mycursor:
    print(x)


mycursor.execute("SELECT * FROM Accuracy, Models WHERE Accuracy.modelID = Models.ID")
for x in mycursor:
    print(x)