import React, { useState, useEffect } from "react";
import { cqt } from "./Cqt";
import * as tf from "@tensorflow/tfjs";

function AudioPlayer() {
  const [audioContext, setAudioContext] = useState(null);
  const [audioBuffer, setAudioBuffer] = useState(null);

  const handleFileChange = async (event) => {
    const files = event.target.files;
    if (files.length > 0) {
      const file = files[0];
      const reader = new FileReader();

      reader.onload = async (e) => {
        const audioData = e.target.result;
        if (!audioContext) {
          const newAudioContext = new (window.AudioContext ||
            window.webkitAudioContext)();
          setAudioContext(newAudioContext);
        }

        try {
          const buffer = await audioContext.decodeAudioData(audioData);
          setAudioBuffer(buffer);

          const spectrogram2 = cqt(
            buffer.getChannelData(0),
            buffer.sampleRate,
            12 * 3 * 9,
            12 * 3,
            256,
            27.5,
            0.5
          );

          const spectrogram = spectrogram2.div(tf.max(spectrogram2));
          tf.max(spectrogram).print();
          tf.min(spectrogram).print();

          const canvas = document.createElement("canvas");
          canvas.width = spectrogram.shape.width;
          canvas.height = spectrogram.shape.height;
          await tf.browser.toPixels(spectrogram.mul(255).cast("int32"), canvas);

          const imageURI = canvas.toDataURL("image/png");
          console.log(imageURI);

          const link = document.createElement("a");
          link.download = "spectrogram.png"; // 保存するファイル名
          link.href = imageURI; // ダウンロードする画像データのURL
          link.click(); // リンクをクリックしてダウンロードを実行

          console.log("Audio buffer is ready!");
        } catch (error) {
          console.error("Error decoding audio data:", error);
        }
      };

      reader.readAsArrayBuffer(file);
    }
  };

  const getAudioArray = () => {
    if (!audioBuffer) {
      console.log("No audio loaded!");
      return;
    }

    // Assuming you want the data from the first channel
    const channelData = audioBuffer.getChannelData(0);
    console.log("Audio Data Array:", channelData);
  };

  return (
    <div>
      <input type="file" onChange={handleFileChange} accept="audio/*" />
      <button onClick={getAudioArray} disabled={!audioBuffer}>
        Get Audio Data Array
      </button>

      <canvas id="myCanvas" width="800" height="600"></canvas>
    </div>
  );
}

export default AudioPlayer;
