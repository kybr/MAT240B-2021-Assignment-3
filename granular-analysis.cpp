// Assignment 3
//
// -- Karl Yerkes / 2021-01-23 / MAT240B
//

#include <algorithm>  // std::sort, std::min
#include <complex>
#include <iostream>
#include <limits>
#include <map>
#include <valarray>
#include <vector>

const double SAMPLERATE = 48000.0;

typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;

// Cooley–Tukey FFT (in-place, divide-and-conquer)
// Higher memory requirements and redundancy although more intuitive
void fft(CArray& x) {
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

struct Peak {
  double magnitude, frequency;
};

struct Grain {
  float peakToPeak;
  float rms;
  float zcr;
  float centroid;
  float f0;

  void show() {
    printf("p2p:%lf rms:%lf zcr:%lf centroid:%lf f0:%lf\n", peakToPeak, rms,
           zcr, centroid, f0);
  }
};

int main(int argc, char* argv[]) {
  std::vector<double> input;
  double value;
  while (std::cin >> value) {
    input.push_back(value);
  }

  int clipSize = 2048;
  int hopSize = 1024;
  int fftSize = 8192;

  std::vector<Grain> grainList;
  for (int n = 0; n < input.size() - clipSize; n += hopSize) {
    std::vector<double> clip(clipSize, 0.0);
    for (int i = 0; i < clip.size(); i++) {
      clip[i] = input[n + i];
    }

    //
    //
    //

    grainList.emplace_back();
    Grain& grain(grainList.back());

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

    //
    double sum = 0;
    for (int i = 0; i < clip.size(); i++) {
      sum += clip[i] * clip[i];
    }
    grain.rms = ::sqrt(sum / clip.size());

    //
    int crossing = 0;
    for (int i = 1; i < clip.size(); i++) {
      if (clip[i - 1] * clip[i] < 0) {
        crossing++;
      }
    }
    grain.zcr = 1.0 * crossing / clip.size() * SAMPLERATE;

    //
    for (int i = 0; i < clip.size(); i++) {
      // windowed copy
      clip[i] = input[n + i] * (1 - cos(2 * M_PI * i / clip.size())) / 2;
    }

    CArray data;
    data.resize(fftSize);
    for (int i = 0; i < clip.size(); i++) {
      data[i] = clip[i];
    }
    for (int i = clip.size(); i < fftSize; i++) {
      data[i] = 0.0;
    }

    fft(data);

    //
    double numerator = 0;
    double denominator = 0;
    for (int i = 0; i < data.size() / 2 + 1; i++) {
      numerator += abs(data[i]) * SAMPLERATE * i / data.size();
      denominator += abs(data[i]);
    }
    grain.centroid = numerator / denominator;

    //
    std::vector<Peak> peak;
    for (int i = 1; i < data.size() / 2; i++) {
      if (abs(data[i - 1]) < abs(data[i]))
        if (abs(data[i + 1]) < abs(data[i]))
          peak.push_back({abs(data[i]) / (clip.size() / 2),
                          1.0 * SAMPLERATE * i / data.size()});
    }

    std::sort(peak.begin(), peak.end(), [](Peak const& a, Peak const& b) {
      return a.magnitude > b.magnitude;
    });

    if (peak.size() < 10) {
      printf("Weird!\n");
      exit(10);
    }

    peak.resize(10);

    std::map<double, int> histogram;
    for (int i = 0; i < peak.size(); i++) {
      for (int j = 1 + i; j < peak.size(); j++) {
        double d = abs(peak[i].frequency - peak[j].frequency);
        d = round(1000 * d) / 1000;
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

    grain.show();

    /*
    std::vector<double> difference;
    for (int i = 0; i < peak.size(); i++) {
      for (int j = 1 + i; j < peak.size(); j++) {
        difference.push_back(abs(peak[i].frequency - peak[j].frequency));
      }
    }

    std::sort(difference.begin(), difference.end());

    for (auto e : difference) {
      printf("%lf ", e);
    }
    printf("\n");
    */
  }

  return 0;
}
