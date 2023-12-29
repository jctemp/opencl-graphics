use bevy::math::Vec3;

pub struct CurveInterpolator {
    anchor: Vec<Vec3>,
}

impl CurveInterpolator {
    pub fn new(x: Vec<f32>, y: Vec<f32>) -> CurveInterpolator {
        let mut anchor = Vec::new();
        for i in 0..x.len() {
            anchor.push(Vec3::new(x[i], y[i], 0.0));
        }
        CurveInterpolator { anchor }
    }

    pub fn interpolate(&self, x: f32, c: Vec<f32>) -> f32 {
        let mut i = 0;
        while i < self.anchor.len() - 1 && x > self.anchor[i + 1].x {
            i += 1;
        };

        // TODO: !!!

        0.0
    }
}

