use bevy::math::Vec3;

pub struct CurveInterpolator;

impl CurveInterpolator {
    fn interpolate(x: &Vec<f32>, c: &Vec<f32>, t: f32) -> f32 {
        let mut i = 1;
        while i < x.len() - 1 && t > x[i] {
            i += 1;
        }

        let x0 = x[i - 1];
        let x1 = x[i];

        let c0 = c[i - 1];
        let c1 = c[i];

        let h = x1 - x0;

        let b = (1.0 / h) * (t - x0) - (h / 6.0) * (c1 - c0);
        let a = x0 + 0.5 * b * h - (1.0 / 6.0) * c0 * h * h;

        let y = (1.0 / (6.0 * h)) * c1 * (t - x0).powi(3)
            + (1.0 / (6.0 * h)) * c0 * (x1 - t)
            + b * (t - 0.5 * (x0 + x1))
            + a;

        y
    }

    pub fn generate(x: &Vec<f32>, c: &Vec<f32>) -> Vec<Vec3> {
        (0..100)
            .map(|i| {
                let min = x[0];
                let max = x[x.len() - 1];
                let t = min + (max - min) * (i as f32 / 100.0);
                let y = Self::interpolate(x, c, t);
                Vec3::new(t, y, 0.0)
            })
            .collect()
    }
}
