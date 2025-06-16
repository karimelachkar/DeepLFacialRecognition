import React, { useState, useEffect } from "react";
import {
  StyleSheet,
  View,
  Button,
  Image,
  Alert,
  Text,
  ActivityIndicator,
  ScrollView,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { Camera } from "expo-camera";

const RENDER_URL = "https://deeplfacialrecognition.onrender.com";

export default function HomeScreen() {
  const [idImage, setIdImage] = useState<string | null>(null);
  const [recognizeImage, setRecognizeImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    (async () => {
      const cameraStatus = await Camera.requestCameraPermissionsAsync();
      if (cameraStatus.status !== "granted") {
        Alert.alert(
          "Permission needed",
          "Sorry, we need camera permissions to make this work!"
        );
      }
    })();
  }, []);

  const pickImage = async (
    setImage: React.Dispatch<React.SetStateAction<string | null>>
  ) => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };

  const takePicture = async (
    setImage: React.Dispatch<React.SetStateAction<string | null>>
  ) => {
    const cameraStatus = await Camera.getCameraPermissionsAsync();
    if (cameraStatus.status !== "granted") {
      Alert.alert("Permission needed", "Camera permission is not granted.");
      return;
    }

    let result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };

  const recognizeFace = async () => {
    if (!idImage || !recognizeImage) {
      Alert.alert(
        "Missing Image",
        "Please provide both an ID image and an image to recognize."
      );
      return;
    }

    setIsLoading(true);

    const formData = new FormData();

    const idUriParts = idImage.split(".");
    const idFileType = idUriParts[idUriParts.length - 1];
    formData.append("file1", {
      uri: idImage,
      name: `id_photo.${idFileType}`,
      type: `image/${idFileType}`,
    } as any);

    const recognizeUriParts = recognizeImage.split(".");
    const recognizeFileType = recognizeUriParts[recognizeUriParts.length - 1];
    formData.append("file2", {
      uri: recognizeImage,
      name: `recognize_photo.${recognizeFileType}`,
      type: `image/${recognizeFileType}`,
    } as any);

    try {
      const response = await fetch(`${RENDER_URL}/verify`, {
        // Assuming the new endpoint is /verify
        method: "POST",
        body: formData,
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      const data = await response.json();

      if (response.ok) {
        Alert.alert(
          "Recognition Result",
          `Match: ${data.verified}\nSimilarity: ${data.similarity.toFixed(2)}%`
        );
      } else {
        Alert.alert(
          "Recognition Failed",
          data.error || "An unknown error occurred."
        );
      }
    } catch (error) {
      console.error(error);
      Alert.alert(
        "Error",
        "An error occurred while communicating with the server."
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>One-Shot Face Recognition</Text>

      <Text style={styles.sectionTitle}>ID Picture</Text>
      <View style={styles.imageContainer}>
        {idImage ? (
          <Image source={{ uri: idImage }} style={styles.image} />
        ) : (
          <Text>No ID image selected</Text>
        )}
      </View>
      <View style={styles.buttonContainer}>
        <Button title="Upload ID" onPress={() => pickImage(setIdImage)} />
        <Button
          title="Take ID Picture"
          onPress={() => takePicture(setIdImage)}
        />
      </View>

      <Text style={styles.sectionTitle}>Image to Recognize</Text>
      <View style={styles.imageContainer}>
        {recognizeImage ? (
          <Image source={{ uri: recognizeImage }} style={styles.image} />
        ) : (
          <Text>No image to recognize</Text>
        )}
      </View>
      <View style={styles.buttonContainer}>
        <Button
          title="Upload Image"
          onPress={() => pickImage(setRecognizeImage)}
        />
        <Button
          title="Take Picture"
          onPress={() => takePicture(setRecognizeImage)}
        />
      </View>

      {isLoading ? (
        <ActivityIndicator
          size="large"
          color="#0000ff"
          style={styles.recognizeButton}
        />
      ) : (
        <View style={styles.recognizeButton}>
          <Button
            title="Recognize Face"
            onPress={recognizeFace}
            disabled={!idImage || !recognizeImage}
          />
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: "600",
    marginTop: 20,
    marginBottom: 10,
  },
  imageContainer: {
    width: 250,
    height: 250,
    borderWidth: 1,
    borderColor: "#ccc",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 10,
  },
  image: {
    width: "100%",
    height: "100%",
  },
  buttonContainer: {
    flexDirection: "row",
    justifyContent: "space-around",
    width: "80%",
    marginBottom: 20,
  },
  recognizeButton: {
    marginTop: 20,
    width: "80%",
  },
});
