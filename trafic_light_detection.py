from roboflow import Roboflow
rf = Roboflow(api_key="JSZlqImP83L1s9dhT5nN")
project = rf.workspace().project("trafic_light")
model = project.version(1).model

# infer on a local image
print(model.predict("test3.jpeg", confidence=40, overlap=30).json())#bu satır kordinatları cıktı olarak verir. ornegin test3 icin cıktısı>>>>"{'predictions': [{'x': 419.5, 'y': 104.0, 'width': 33.0, 'height': 70.0, 'confidence': 0.7646297812461853, 'class': 'car', 'class_id': 0, 'image_path': 'test3.jpeg', 'prediction_type': 'ObjectDetectionModel'}, {'x': 132.0, 'y': 299.5, 'width': 50.0, 'height': 75.0, 'confidence': 0.7629860639572144, 'class': 'car', 'class_id': 0, 'image_path': 'test3.jpeg', 'prediction_type': 'ObjectDetectionModel'}, {'x': 230.0, 'y': 276.5, 'width': 22.0, 'height': 25.0, 'confidence': 0.4688820540904999, 'class': 'car', 'class_id': 0, 'image_path': 'test3.jpeg', 'prediction_type': 'ObjectDetectionModel'}], 'image': {'width': '474', 'height': '474'}}"

# visualize your prediction
model.predict("test3.jpeg", confidence=40, overlap=30).save("prediction.jpg")# bu satırda ise cıktı goruntudur ve prediction adlı dosyada kaydedilir.

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
