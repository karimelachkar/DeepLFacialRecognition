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
  TextInput,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { Camera } from "expo-camera";

const RENDER_URL = "https://deeplfacialrecognition.onrender.com";

export default function HomeScreen() {
  const [idImage, setIdImage] = useState<string | null>(null);
  const [idName, setIdName] = useState("");
  const [recognizeImage, setRecognizeImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [feedback, setFeedback] = useState("");

  useEffect(() => {
    (async () => {
      await Camera.requestCameraPermissionsAsync();
    })();
  }, []);

  const pickImage = async (
    setImage: React.Dispatch<React.SetStateAction<string | null>>
  ) => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: "images",
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
    const { status } = await Camera.getCameraPermissionsAsync();
    if (status !== "granted") {
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

  const handleRegister = async () => {
    if (!idImage || !idName) {
      Alert.alert(
        "Missing Information",
        "Please provide a name and an ID image."
      );
      return;
    }
    setIsLoading(true);
    setFeedback("Registering...");
    const formData = new FormData();
    const uriParts = idImage.split(".");
    const fileType = uriParts[uriParts.length - 1];
    formData.append("image", {
      uri: idImage,
      name: `photo.${fileType}`,
      type: `image/${fileType}`,
    } as any);
    formData.append("name", idName);

    try {
      const response = await fetch(`${RENDER_URL}/register`, {
        method: "POST",
        body: formData,
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      const data = await response.json();
      if (response.ok) {
        setFeedback(`Successfully registered ${data.name}!`);
        Alert.alert("Success", `Successfully registered ${data.name}!`);
      } else {
        throw new Error(data.error || "Registration failed");
      }
    } catch (error: any) {
      setFeedback(`Registration failed: ${error.message}`);
      Alert.alert("Error", `Registration failed: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRecognize = async () => {
    if (!recognizeImage) {
      Alert.alert("Missing Image", "Please provide an image to recognize.");
      return;
    }
    setIsLoading(true);
    setFeedback("Recognizing...");
    const formData = new FormData();
    const uriParts = recognizeImage.split(".");
    const fileType = uriParts[uriParts.length - 1];
    formData.append("image", {
      uri: recognizeImage,
      name: `photo.${fileType}`,
      type: `image/${fileType}`,
    } as any);

    try {
      const response = await fetch(`${RENDER_URL}/recognize`, {
        method: "POST",
        body: formData,
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      const data = await response.json();
      if (response.ok) {
        const resultText = `Recognized: ${data.name || "Unknown"}`;
        setFeedback(resultText);
        Alert.alert("Recognition Result", resultText);
      } else {
        throw new Error(data.error || "Recognition failed");
      }
    } catch (error: any) {
      setFeedback(`Recognition failed: ${error.message}`);
      Alert.alert("Error", `Recognition failed: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Dynamic Face Recognition</Text>
      <Text style={styles.feedbackText}>{feedback}</Text>

      {/* Registration Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Step 1: Register an ID</Text>
        <TextInput
          style={styles.input}
          placeholder="Enter Name"
          value={idName}
          onChangeText={setIdName}
        />
        <View style={styles.imageContainer}>
          {idImage ? (
            <Image source={{ uri: idImage }} style={styles.image} />
          ) : (
            <Text>No ID image</Text>
          )}
        </View>
        <View style={styles.buttonContainer}>
          <Button title="Upload ID" onPress={() => pickImage(setIdImage)} />
          <Button
            title="Take Picture"
            onPress={() => takePicture(setIdImage)}
          />
        </View>
        <Button
          title="Register"
          onPress={handleRegister}
          disabled={isLoading || !idImage || !idName}
        />
      </View>

      {/* Recognition Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Step 2: Recognize a Face</Text>
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
        <Button
          title="Recognize"
          onPress={handleRecognize}
          disabled={isLoading || !recognizeImage}
        />
      </View>

      {isLoading && (
        <ActivityIndicator
          size="large"
          color="#0000ff"
          style={styles.loading}
        />
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: "center",
    paddingVertical: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 10,
  },
  feedbackText: {
    fontSize: 16,
    color: "blue",
    marginBottom: 20,
    height: 20,
  },
  section: {
    width: "90%",
    borderWidth: 1,
    borderColor: "#ddd",
    borderRadius: 8,
    padding: 15,
    marginBottom: 25,
    alignItems: "center",
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: "600",
    marginBottom: 15,
  },
  input: {
    height: 40,
    borderColor: "gray",
    borderWidth: 1,
    width: "100%",
    marginBottom: 15,
    paddingHorizontal: 10,
    borderRadius: 5,
  },
  imageContainer: {
    width: 200,
    height: 200,
    borderWidth: 1,
    borderColor: "#ccc",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 15,
    backgroundColor: "#f0f0f0",
  },
  image: {
    width: "100%",
    height: "100%",
  },
  buttonContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
    width: "100%",
    marginBottom: 15,
  },
  loading: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: [{ translateX: -25 }, { translateY: -25 }],
  },
});
