import * as tf from "@tensorflow/tfjs";

class IIRFilter {
  constructor(b, a) {
    this.b = b; // 分子係数（入力の影響）
    this.a = a; // 分母係数（過去の出力の影響）
    this.inputHistory = new Array(b.length).fill(0);
    this.outputHistory = new Array(a.length).fill(0);
  }

  process(input) {
    // 入力履歴を更新
    this.inputHistory.unshift(input);
    this.inputHistory.pop();

    // 出力の計算
    let output = 0.0;
    for (let i = 0; i < this.b.length; i++) {
      output += this.b[i] * this.inputHistory[i];
    }
    for (let i = 1; i < this.a.length; i++) {
      output -= this.a[i] * this.outputHistory[i];
    }

    // 出力履歴を更新
    this.outputHistory.unshift(output);
    this.outputHistory.pop();

    return output;
  }
}

function lowPassFilter(signal) {
  const b = [
    0.02321932, 0.13931594, 0.34828986, 0.46438647, 0.34828986, 0.13931594,
    0.02321932,
  ];
  const a = [
    1.0, 3.02225963e-2, 4.46204537e-1, -2.76669843e-2, 3.94304556e-2,
    -2.55561209e-3, 4.01725661e-4,
  ];
  const filter = new IIRFilter(b, a);

  const filteredSignal = signal.map((sample) => filter.process(sample));
  return filteredSignal;
}

function zeroPhaseFilter(signal) {
  // 前向きフィルタリング
  const forwardFiltered = lowPassFilter(signal);

  // 信号を逆転
  const reversedSignal = forwardFiltered.slice().reverse();

  // 逆向きフィルタリング
  const reverseFiltered = lowPassFilter(reversedSignal);

  // 最終的な信号を正しい順序に戻す
  return reverseFiltered.reverse();
}

function getHanningWindow(M) {
  const x = new Array(M);
  for (let i = 0; i < M; ++i) {
    x[i] = 0.5 - 0.5 * Math.cos((Math.PI * 2 * i) / (M - 1));
  }

  return x;
}

function getCqtFrequencies(nBins, fMin, binsPerOctave) {
  const freqs = new Array(nBins);
  for (let i = 0; i < nBins; ++i) {
    freqs[i] = fMin * Math.pow(2.0, i / binsPerOctave);
  }
  return freqs;
}

function conjugate(tensor) {
  return tf.tidy(() => {
    const real = tf.real(tensor);
    const imag = tf.imag(tensor);

    return tf.complex(real, imag.mul(-1));
  });
}

function divComplex(tensor, number) {
  return tf.tidy(() => {
    const real = tf.real(tensor);
    const imag = tf.imag(tensor);

    return tf.complex(real.div(number), imag.div(number));
  });
}

function matmulComplex(A, B) {
  const Ar = tf.real(A);
  const Ai = tf.imag(A);
  const Br = tf.real(B);
  const Bi = tf.imag(B);

  const temp1 = tf.matMul(Ar, Br);
  const temp2 = tf.matMul(Ai, Bi);
  const temp3 = tf.matMul(Ar, Bi);
  const temp4 = tf.matMul(Ai, Br);

  const Cr = temp1.sub(temp2);
  const Ci = temp3.add(temp4);

  Ar.dispose();
  Ai.dispose();
  Br.dispose();
  Bi.dispose();
  temp1.dispose();
  temp2.dispose();
  temp3.dispose();
  temp4.dispose();

  return tf.complex(Cr, Ci);
}

function getSpectralKernel(
  samplingRate,
  nBins,
  binsPerOctave,
  qValue,
  nFFT1Octave,
  cqtFreqs
) {
  const tempKernelsReal = [];
  const tempKernelsImag = [];
  for (let k = nBins - binsPerOctave; k < nBins; k++) {
    const freq = cqtFreqs[k];
    const nK = parseInt(Math.ceil((samplingRate * qValue) / freq));
    const startWin = parseInt((nFFT1Octave - nK) / 2);

    let tempKernelReal = new Array(nFFT1Octave).fill(0);
    let tempKernelImag = new Array(nFFT1Octave).fill(0);
    const hanningWindow = getHanningWindow(nK);

    for (let i = 0; i < nK; ++i) {
      tempKernelReal[startWin + i] =
        Math.cos(Math.PI * 2 * (freq / samplingRate) * i) *
        (hanningWindow[i] / nK);
      tempKernelImag[startWin + i] =
        Math.sin(Math.PI * 2 * (freq / samplingRate) * i) *
        (hanningWindow[i] / nK);
    }

    tempKernelsReal.push(tempKernelReal);
    tempKernelsImag.push(tempKernelImag);
  }

  const realParts = tf.tensor2d(
    tempKernelsReal,
    [binsPerOctave, nFFT1Octave],
    "float32"
  );
  const imagParts = tf.tensor2d(
    tempKernelsImag,
    [binsPerOctave, nFFT1Octave],
    "float32"
  );

  const complexTensor = tf.complex(realParts, imagParts);
  realParts.dispose();
  imagParts.dispose();

  const fftResults = tf.fft(complexTensor);
  complexTensor.dispose();
  return conjugate(fftResults);
}

export function cqt(
  signals,
  samplingRate,
  nBins,
  binsPerOctave,
  hopLength,
  fMin,
  qFactor
) {
  const freqs = getCqtFrequencies(nBins, fMin, binsPerOctave);
  const fRatio = 1.0 / binsPerOctave;
  const qValue = (1.0 / (Math.pow(2.0, fRatio) - 1)) * qFactor;
  const nFrame = parseInt(signals.length / hopLength);
  const nFFTExp =
    Math.floor(
      Math.log2(parseInt(Math.ceil((samplingRate * qValue) / freqs[0])))
    ) + 1;
  const nFFT = parseInt(Math.pow(2.0, nFFTExp));
  const nOctave = nBins / binsPerOctave;
  const nFFT1Octave = parseInt(Math.pow(2.0, nFFTExp - (nOctave - 1)));

  // カーネル行列計算
  let spectralKernel = getSpectralKernel(
    samplingRate,
    nBins,
    binsPerOctave,
    qValue,
    nFFT1Octave,
    freqs
  );
  spectralKernel = divComplex(spectralKernel, nFFT);

  const nXAdjust = Math.pow(2.0, nOctave - 1);
  const nPadding = nXAdjust - ((signals.length + nFFT) % nXAdjust);
  const newSignalsLength = signals.length + nFFT + nPadding;

  // 再帰ダウンサンプリング
  const resampledSignals = new Array(nOctave);
  resampledSignals[0] = new Array(newSignalsLength).fill(0);
  for (let i = 0; i < signals.length; ++i) {
    resampledSignals[0][i] = signals[i];
  }
  for (let k = 1; k < nOctave; ++k) {
    const prevResampledLength = parseInt(
      newSignalsLength / Math.pow(2.0, k - 1)
    );
    const tempSignals = new Array(prevResampledLength);
    for (let i = 0; i < prevResampledLength; ++i) {
      tempSignals[i] = resampledSignals[k - 1][i];
    }

    const filteredSignals = zeroPhaseFilter(tempSignals);

    const resampledLength = prevResampledLength / 2;
    resampledSignals[k] = new Array(newSignalsLength).fill(0);
    for (let i = 0; i < resampledLength; ++i) {
      resampledSignals[k][i] = filteredSignals[i * 2];
    }
  }

  // CQTSpectrogramを計算
  const resultSpectrogram = [];
  for (let k = 0; k < nOctave; ++k) {
    const centerInit = nFFT / Math.pow(2.0, k + 1);
    const hopLengthDs = hopLength / Math.pow(2.0, k);

    const frames = [];
    for (let n = 0; n < nFrame; ++n) {
      const center = centerInit + n * hopLengthDs;
      const startWin = center - nFFT1Octave / 2;

      const tempFrame = new Array(nFFT1Octave);
      for (let i = 0; i < nFFT1Octave; ++i) {
        tempFrame[i] = resampledSignals[k][startWin + i];
      }
      frames.push(tempFrame);
    }

    const realParts = tf.tensor2d(frames, [nFrame, nFFT1Octave], "float32");
    const imagParts = tf.zeros([nFrame, nFFT1Octave], "float32");
    const complexTensor = tf.complex(realParts, imagParts);
    realParts.dispose();
    imagParts.dispose();

    const fftResults = tf.fft(complexTensor);
    const fftResultsTranspose = tf.transpose(fftResults, [1, 0]);
    fftResults.dispose();

    const spectrogram = matmulComplex(
      spectralKernel,
      fftResultsTranspose
    ).abs();
    resultSpectrogram.unshift(spectrogram);

    fftResultsTranspose.dispose();
    complexTensor.dispose();
  }

  spectralKernel.dispose();
  return tf.concat(resultSpectrogram, 0);
}