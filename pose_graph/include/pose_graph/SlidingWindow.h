#include <deque>

template <typename T>
class SlidingWindow {
 public:
  SlidingWindow(int window_size = 0) : window_size_(window_size){};
  ~SlidingWindow() = default;

  void push(const T& element) {
    if (window_.size() == window_size_) {
      sum_ -= window_.front();
      window_.pop_front();
    }
    window_.push_back(element);
    sum_ += element;
  }

  void set_windowsize(const size_t& window_size) { window_size_ = window_size; }

  const std::deque<T>& get_window() const { return window_; }

  T get_average() const { return sum_ / window_.size(); }

 private:
  std::deque<T> sliding_window_;
  T sum_{};
  int window_size_;
};