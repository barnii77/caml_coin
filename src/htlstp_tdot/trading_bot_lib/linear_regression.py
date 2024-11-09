from typing import List


class LinearRegression:
    def __init__(
        self,
        x_values: List[float],
        y_values: List[float],
    ):
        x_mean = self.mean(x_values)
        y_mean = self.mean(y_values)
        self.slope = self.covariance(
            x_values, x_mean, y_values, y_mean
        ) / self.variance(x_values, x_mean)
        self.intercept = y_mean - self.slope * x_mean

    @staticmethod
    def mean(values):
        return sum(values) / len(values)

    @staticmethod
    def variance(values, mean_value):
        return sum((x - mean_value) ** 2 for x in values)

    @staticmethod
    def covariance(x_values, x_mean, y_values, y_mean):
        return sum(
            (x_values[i] - x_mean) * (y_values[i] - y_mean)
            for i in range(len(x_values))
        )

    def predict(self, x_values):
        if isinstance(x_values, (int, float)):
            return self.intercept + self.slope * x_values
        return [self.intercept + self.slope * x for x in x_values]


if __name__ == "__main__":
    model = LinearRegression([1, 2, 3], [-2, -2.3, -3.2])
    print(f"Intercept (b0): {model.intercept:.2f}")
    print(f"Slope (b1): {model.slope:.2f}")

    predictions = model.predict(4)
    print("Predictions:", predictions)
