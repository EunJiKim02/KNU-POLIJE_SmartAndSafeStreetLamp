import firebase_admin
from firebase_admin import db, credentials
from firebase_admin import storage

import time

cred = credentials.Certificate("esp32-b8e97-firebase-adminsdk-gtaqd-34a681d2f0.json")

firebase_admin.initialize_app(cred, {
    "storageBucket" : "esp32-b8e97.appspot.com",
    "databaseURL" : "https://esp32-b8e97-default-rtdb.asia-southeast1.firebasedatabase.app/"
    })
ref = db.reference('/data')
data = {
  "10101010":
    {
        "info" : "just test in python",
        "image": "link", 
        "time" : "2023-08-22" 
    }
}
db.reference('/data').update(data)

bucket = storage.bucket()


blob = bucket.blob("Screen.png")
blob.upload_from_filename("Screen.png")

<<<<<<< HEAD
url = blob.generate_signed_url()
=======

>>>>>>> 83c960978bea6c300aa0a393199b1e8d6e96e22d
