from change_analysis.v2_inference import V2ChangeDetector


class V2DetectorAdapter:

    def __init__(self):

        self.detector = (
            V2ChangeDetector()
        )

    def __call__(
        self,
        image_before,
        image_after,
    ):

        return self.detector.predict(
            image_before,
            image_after
        )
