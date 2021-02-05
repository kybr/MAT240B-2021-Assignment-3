// Assignment 3
//
//
// -- Karl Yerkes / 2021-01-23 / MAT240B
//
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>  // std::sort
#include <algorithm>  // std::sort, std::min
#include <cmath>      // ::cos()
#include <complex>
#include <iostream>
#include <valarray>
#include <vector>

#include "al/app/al_App.hpp"
#include "al/ui/al_ControlGUI.hpp"
#include "al/ui/al_Parameter.hpp"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;

const int BLOCK_SIZE = 512;
const int SAMPLE_RATE = 48000;

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

double dbtoa(double db) { return pow(10.0, db / 20.0); }

// Cooley–Tukey FFT (in-place, divide-and-conquer)
// Higher memory requirements and redundancy although more intuitive
void fft(CArray &x) {
  const size_t N = x.size();
  if (N <= 1) return;

  // divide
  CArray even = x[std::slice(0, N / 2, 2)];
  CArray odd = x[std::slice(1, N / 2, 2)];

  // conquer
  fft(even);
  fft(odd);

  // combine
  for (size_t k = 0; k < N / 2; ++k) {
    Complex t = std::polar(1.0, -2 * M_PI * k / N) * odd[k];
    x[k] = even[k] + t;
    x[k + N / 2] = even[k] - t;
  }
}

void load(std::vector<float> &input, const char *filePath) {
  unsigned int channels;
  unsigned int sampleRate;
  drwav_uint64 totalPCMFrameCount;
  float *pSampleData = drwav_open_file_and_read_pcm_frames_f32(
      filePath, &channels, &sampleRate, &totalPCMFrameCount, NULL);
  if (pSampleData == NULL) {
    printf("failed to load %s\n", filePath);
    exit(1);
  }

  //
  if (channels == 1)
    for (int i = 0; i < totalPCMFrameCount; i++) {
      input.push_back(pSampleData[i]);
    }
  else if (channels == 2) {
    for (int i = 0; i < totalPCMFrameCount; i++) {
      input.push_back((pSampleData[2 * i] + pSampleData[2 * i + 1]) / 2);
    }
  } else {
    printf("can't handle %d channels\n", channels);
    exit(1);
  }

  drwav_free(pSampleData, NULL);
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

struct Peak {
  double magnitude, frequency;
};

struct Counter {
  int i{0};
  int operator()(int high, int low = 0) {
    int value = i;

    // side effect
    i++;
    if (i >= high) {
      i = low;
    }

    return value;
  }
};

struct Timer {
  int i{0};

  bool operator()(int limit) {
    bool value = false;

    i++;
    if (i >= limit) {
      i = 0;
      value = true;
    }

    return value;
  }
};

struct Grain {
  int begin;  // index into the vector<float> of the sound file
  int end;    // index into the vector<float> of the sound file

  float peakToPeak;
  float rms;
  float zcr;
  float centroid;
  float f0;

  // play back a windowed version of this grain
  //
  Counter counter;
  float operator()(std::vector<float> const &audio) {
    int i = counter(end - begin);
    return audio[i] * (1 - cos(2 * M_PI * i / (end - begin))) / 2;
  }
};

struct GrainPlayer {
  Timer advance;
  int index{0};

  float next(std::vector<Grain> const &grain, std::vector<float> const &audio) {
    int length = grain.size();

    if (advance(2048)) {
      index++;
      if (index >= length) {
        index = 0;
      }
    }

    int a = index - 1;
    int b = index;
    int c = index + 1;

    // fix up
    if (a < 0) {
      a += length;
    }
    if (c >= length) {
      c -= length;
    }

    return grain[a](audio) + grain[b](audio) + grain[c](audio);
  }
};

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

using namespace al;

struct MyApp : App {
  Parameter background{"background", "", 0.0, "", 0.0f, 1.0f};
  Parameter db{"db", "", -60.0, "", -60.0f, 0.0f};
  ControlGUI gui;

  std::vector<float> input;
  std::vector<Grain> grain;
  GrainPlayer player;

  MyApp(int argc, char *argv[]) {
    if (argc < 2) {
      printf("granular-synth <.wav>");
      exit(1);
    }

    load(input, argv[1]);
    printf("input size is %ld\n", input.size());
    fflush(stdout);

    // this is how to get a command line argument
    if (argc > 2) {
      int N = std::stoi(argv[2]);
    }

    int clipSize = 2048;
    int hopSize = 1024;
    int fftSize = 8192;

    for (int n = 0; n < input.size() - clipSize; n += hopSize) {
      Grain grain;

      grain.begin = n;
      grain.end = n + clipSize;

      std::vector<double> clip(clipSize, 0.0);
      for (int i = 0; i < clip.size(); i++) {
        //          input      *       Hann window
        clip[i] = input[n + i] * (1 - cos(2 * M_PI * i / clip.size())) / 2;
      }

      // peak-to-peak
      //
      double maximum = std::numeric_limits<double>::min();
      double minimum = std::numeric_limits<double>::max();
      for (int i = 0; i < clip.size(); i++) {
        if (clip[i] > maximum) {
          maximum = clip[i];
        }
        if (clip[i] < minimum) {
          minimum = clip[i];
        }
      }
      grain.peakToPeak = maximum - minimum;

      // root mean squared
      //
      double sum = 0;
      for (int i = 0; i < clip.size(); i++) {
        sum += clip[i] * clip[i];
      }
      grain.rms = ::sqrt(sum / clip.size());

      // zero crossing rate
      //
      int crossing = 0;
      for (int i = 1; i < clip.size(); i++) {
        if (clip[i - 1] * clip[i] < 0) {
          crossing++;
        }
      }
      grain.zcr = 1.0 * crossing / clip.size() * SAMPLE_RATE;

      CArray data;
      data.resize(fftSize);
      for (int i = 0; i < clip.size(); i++) {
        data[i] = clip[i];
      }
      for (int i = clip.size(); i < fftSize; i++) {
        data[i] = 0.0;
      }

      // XXX the size of data really must be a power of two!
      //
      fft(data);

      // spectral centroid
      //
      double numerator = 0;
      double denominator = 0;
      for (int i = 0; i < data.size() / 2 + 1; i++) {
        numerator += abs(data[i]) * SAMPLE_RATE * i / data.size();
        denominator += abs(data[i]);
      }
      grain.centroid = numerator / denominator;

      std::vector<Peak> peak;
      for (int i = 1; i < data.size() / 2; i++) {
        // only accept maxima
        //
        if (abs(data[i - 1]) < abs(data[i]))
          if (abs(data[i + 1]) < abs(data[i]))
            peak.push_back({abs(data[i]) / (clip.size() / 2),
                            1.0 * SAMPLE_RATE * i / data.size()});
      }

      std::sort(peak.begin(), peak.end(), [](Peak const &a, Peak const &b) {
        return a.magnitude > b.magnitude;
      });

      peak.resize(10);  // throw away the extras

      // f0 estimation
      //
      std::map<double, int> histogram;
      for (int i = 0; i < peak.size(); i++) {
        for (int j = 1 + i; j < peak.size(); j++) {
          double d = abs(peak[i].frequency - peak[j].frequency);
          const double factor = 100;
          d = round(factor * d) / factor;
          if (histogram.count(d)) {
            histogram[d]++;
          } else {
            histogram[d] = 1;
          }
        }
      }

      int max = 0;
      grain.f0 = 0;
      for (auto e : histogram) {
        if (e.second > max) {
          max = e.second;
          grain.f0 = e.first;
        }
      }

      /*
      typedef std::pair<double, int> T;
      std::vector<T> foo(histogram.begin(), histogram.end());
      std::sort(foo.begin(), foo.end(),
                [](T const &a, T const &b) { return a.second > b.second; });

      for (auto e : foo) {
        printf("%lf %d\n", e.first, e.second);
      }
      printf("\n");
      */

      this->grain.emplace_back(grain);
    }
  }

  void onCreate() override {
    gui << background;
    gui << db;
    gui.init();
    navControl().active(false);
  }

  void onAnimate(double dt) override {
    //
  }

  void onDraw(Graphics &g) override {
    g.clear(background);
    gui.draw(g);
  }

  void onSound(AudioIOData &io) override {
    while (io()) {
      float f = 0;

      // XXX
      f += player.next(grain, audio);
      //

      f *= dbtoa(db.get());
      io.out(0) = f;
      io.out(1) = f;
    }
  }

  bool onKeyDown(const Keyboard &k) override {
    int midiNote = asciiToMIDI(k.key());

    switch (k.key()) {
      case '0':
        sort(grain.begin(), grain.end(),
             [](Grain const &a, Grain const &b) { return a.begin < b.begin; });
        break;

      case '1':
        sort(grain.begin(), grain.end(), [](Grain const &a, Grain const &b) {
          return a.peakToPeak < b.peakToPeak;
        });
        break;

      case '2':
        sort(grain.begin(), grain.end(),
             [](Grain const &a, Grain const &b) { return a.rms < b.rms; });
        break;

      case '3':
        sort(grain.begin(), grain.end(),
             [](Grain const &a, Grain const &b) { return a.zcr < b.zcr; });
        break;

      case '4':
        sort(grain.begin(), grain.end(), [](Grain const &a, Grain const &b) {
          return a.centroid < b.centroid;
        });
        break;

      case '5':
        sort(grain.begin(), grain.end(),
             [](Grain const &a, Grain const &b) { return a.f0 < b.f0; });
        break;
    }

    return true;
  }

  bool onKeyUp(const Keyboard &k) override {
    int midiNote = asciiToMIDI(k.key());
    return true;
  }
};

int main(int argc, char *argv[]) {
  MyApp app(argc, argv);
  app.configureAudio(SAMPLE_RATE, BLOCK_SIZE, 2, 1);
  app.start();
  return 0;
}
