//! `oura control` — turn ring motion into laptop input.
//!
//! Consumes the live ACM stream, derives tilt (pitch/roll against a neutral
//! pose calibrated over the first 3 s) and shake gestures, and maps them to
//! macOS key events via `osascript` / System Events. Run with `--dry-run`
//! first to watch gestures fire without sending any keys.
//!
//! Default mapping:
//! - tilt right / left  → right / left arrow (slides, seek)
//! - tilt up / down     → up / down arrow
//! - double shake       → spacebar (play/pause in most apps)
//! - triple shake       → lock screen (ctrl+cmd+Q)
//!
//! Sending key events needs Accessibility permission for the terminal
//! (System Settings → Privacy & Security → Accessibility).

use std::process::Command as Process;
use std::time::{Duration, Instant};

use anyhow::Result;

use oura_link::ble::BleTransport;
use oura_link::client::AcmSample;
use oura_link::OuraClient;

/// Raw accelerometer counts per g (empirical: ~1000 at rest).
const COUNTS_PER_G: f64 = 1000.0;

/// Seconds of stillness used to capture the neutral hand pose.
const CALIBRATION_SECS: f64 = 3.0;

/// Minimum time between any two fired gestures.
const REFRACTORY: Duration = Duration::from_millis(900);

/// How long a tilt must be held past the threshold before it fires.
const TILT_DWELL: Duration = Duration::from_millis(250);

/// Shake spikes further apart than this window belong to separate gestures.
const SHAKE_WINDOW: Duration = Duration::from_millis(900);

/// Quiet time after the last spike before a shake gesture is classified.
const SHAKE_SETTLE: Duration = Duration::from_millis(350);

/// Minimum spacing between two counted shake spikes.
const SPIKE_GAP: Duration = Duration::from_millis(120);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gesture {
    TiltLeft,
    TiltRight,
    TiltUp,
    TiltDown,
    DoubleShake,
    TripleShake,
}

impl Gesture {
    fn label(self) -> &'static str {
        match self {
            Gesture::TiltLeft => "tilt-left  → left arrow",
            Gesture::TiltRight => "tilt-right → right arrow",
            Gesture::TiltUp => "tilt-up    → up arrow",
            Gesture::TiltDown => "tilt-down  → down arrow",
            Gesture::DoubleShake => "double-shake → spacebar",
            Gesture::TripleShake => "triple-shake → lock screen",
        }
    }

    /// AppleScript to perform this gesture's action.
    fn script(self) -> &'static str {
        match self {
            Gesture::TiltLeft => r#"tell application "System Events" to key code 123"#,
            Gesture::TiltRight => r#"tell application "System Events" to key code 124"#,
            Gesture::TiltUp => r#"tell application "System Events" to key code 126"#,
            Gesture::TiltDown => r#"tell application "System Events" to key code 125"#,
            Gesture::DoubleShake => r#"tell application "System Events" to key code 49"#,
            Gesture::TripleShake => {
                r#"tell application "System Events" to keystroke "q" using {control down, command down}"#
            }
        }
    }
}

/// One tilt axis (pitch or roll): threshold + dwell + re-arm hysteresis.
struct TiltAxis {
    fire_deg: f64,
    /// Ready to fire again (delta returned inside the release band).
    armed: bool,
    /// When the current past-threshold excursion started, and its sign.
    dwell_since: Option<(Instant, i8)>,
}

impl TiltAxis {
    fn new(fire_deg: f64) -> Self {
        Self { fire_deg, armed: true, dwell_since: None }
    }

    /// Feed the current delta-from-neutral; `steady` is false while the hand is
    /// shaking (tilt must not fire off shake motion). Returns -1/+1 on fire.
    fn update(&mut self, delta_deg: f64, steady: bool, now: Instant) -> Option<i8> {
        let release = self.fire_deg * 0.5;
        if delta_deg.abs() < release {
            self.armed = true;
        }
        if delta_deg.abs() < self.fire_deg || !steady {
            self.dwell_since = None;
            return None;
        }
        if !self.armed {
            return None;
        }
        let sign = if delta_deg > 0.0 { 1i8 } else { -1i8 };
        match self.dwell_since {
            Some((since, s)) if s == sign => {
                if now.duration_since(since) >= TILT_DWELL {
                    self.armed = false;
                    self.dwell_since = None;
                    return Some(sign);
                }
            }
            _ => self.dwell_since = Some((now, sign)),
        }
        None
    }
}

/// Streaming gesture detector: gravity lowpass → neutral pose → tilt + shake.
pub struct GestureEngine {
    shake_g: f64,
    /// Lowpass-filtered gravity vector (raw counts).
    gravity: [f64; 3],
    samples_seen: u64,
    started: Instant,
    /// Accumulated gravity during calibration; neutral (pitch, roll) after.
    calib_sum: [f64; 3],
    calib_count: u64,
    neutral: Option<(f64, f64)>,
    pitch: TiltAxis,
    roll: TiltAxis,
    spikes: Vec<Instant>,
    last_fire: Option<Instant>,
}

impl GestureEngine {
    pub fn new(tilt_deg: f64, shake_g: f64, now: Instant) -> Self {
        Self {
            shake_g,
            gravity: [0.0, 0.0, 0.0],
            samples_seen: 0,
            started: now,
            calib_sum: [0.0, 0.0, 0.0],
            calib_count: 0,
            neutral: None,
            pitch: TiltAxis::new(tilt_deg),
            roll: TiltAxis::new(tilt_deg),
            spikes: Vec::new(),
            last_fire: None,
        }
    }

    pub fn calibrated(&self) -> bool {
        self.neutral.is_some()
    }

    fn angles(v: &[f64; 3]) -> (f64, f64) {
        let pitch = (-v[0]).atan2((v[1] * v[1] + v[2] * v[2]).sqrt()).to_degrees();
        let roll = v[1].atan2(v[2]).to_degrees();
        (pitch, roll)
    }

    /// Smallest signed angle difference a-b, wrapped to [-180, 180].
    fn wrap(a: f64, b: f64) -> f64 {
        let mut d = a - b;
        while d > 180.0 {
            d -= 360.0;
        }
        while d < -180.0 {
            d += 360.0;
        }
        d
    }

    pub fn on_sample(&mut self, s: AcmSample, now: Instant) -> Option<Gesture> {
        let v = [s.x as f64, s.y as f64, s.z as f64];

        // Gravity lowpass (~0.4 s time constant at 50 Hz); seed on first sample.
        self.samples_seen += 1;
        if self.samples_seen == 1 {
            self.gravity = v;
        } else {
            const ALPHA: f64 = 0.05;
            for i in 0..3 {
                self.gravity[i] += ALPHA * (v[i] - self.gravity[i]);
            }
        }

        // Deviation from gravity, in g: the "how hard is the hand moving" signal.
        let dev_g = ((v[0] - self.gravity[0]).powi(2)
            + (v[1] - self.gravity[1]).powi(2)
            + (v[2] - self.gravity[2]).powi(2))
        .sqrt()
            / COUNTS_PER_G;

        // Calibration: average gravity over the first CALIBRATION_SECS.
        if self.neutral.is_none() {
            for i in 0..3 {
                self.calib_sum[i] += self.gravity[i];
            }
            self.calib_count += 1;
            if now.duration_since(self.started).as_secs_f64() >= CALIBRATION_SECS {
                let mean = [
                    self.calib_sum[0] / self.calib_count as f64,
                    self.calib_sum[1] / self.calib_count as f64,
                    self.calib_sum[2] / self.calib_count as f64,
                ];
                self.neutral = Some(Self::angles(&mean));
            }
            return None;
        }

        let in_refractory = self
            .last_fire
            .is_some_and(|t| now.duration_since(t) < REFRACTORY);

        // Shake: count deviation spikes, classify once the hand settles.
        if dev_g > self.shake_g
            && self.spikes.last().is_none_or(|t| now.duration_since(*t) >= SPIKE_GAP)
        {
            self.spikes.push(now);
        }
        self.spikes
            .retain(|t| now.duration_since(*t) <= SHAKE_WINDOW);
        if let Some(last) = self.spikes.last().copied() {
            if now.duration_since(last) >= SHAKE_SETTLE {
                let count = self.spikes.len();
                self.spikes.clear();
                if count >= 2 && !in_refractory {
                    self.last_fire = Some(now);
                    return Some(if count >= 3 {
                        Gesture::TripleShake
                    } else {
                        Gesture::DoubleShake
                    });
                }
            }
        }

        // Tilt: pitch/roll of the gravity vector, relative to the neutral pose.
        let (np, nr) = self.neutral.unwrap();
        let (p, r) = Self::angles(&self.gravity);
        let steady = dev_g < 0.35 && !in_refractory;
        if let Some(sign) = self.roll.update(Self::wrap(r, nr), steady, now) {
            self.last_fire = Some(now);
            return Some(if sign > 0 { Gesture::TiltRight } else { Gesture::TiltLeft });
        }
        if let Some(sign) = self.pitch.update(Self::wrap(p, np), steady, now) {
            self.last_fire = Some(now);
            return Some(if sign > 0 { Gesture::TiltUp } else { Gesture::TiltDown });
        }
        None
    }
}

fn perform(gesture: Gesture, dry_run: bool) {
    println!("  ▶ {}{}", gesture.label(), if dry_run { "  [dry-run]" } else { "" });
    if dry_run {
        return;
    }
    // Fire-and-forget so a slow osascript never stalls the BLE stream loop.
    let _ = Process::new("osascript").args(["-e", gesture.script()]).spawn();
}

pub async fn run(
    client: OuraClient<BleTransport>,
    seconds: u64,
    tilt_deg: f64,
    shake_g: f64,
    dry_run: bool,
) -> Result<()> {
    println!("Ring control for {seconds}s — hold your hand STILL for 3s to set neutral…");
    if !dry_run {
        println!("(key events need Accessibility permission for this terminal)");
    }

    let mut engine = GestureEngine::new(tilt_deg, shake_g, Instant::now());
    let mut announced = false;
    client
        .stream_accelerometer(Duration::from_secs(seconds), |s| {
            let now = Instant::now();
            let fired = engine.on_sample(s, now);
            if !announced && engine.calibrated() {
                announced = true;
                println!("Calibrated. Tilt = arrows, double-shake = space, triple-shake = lock.");
            }
            if let Some(g) = fired {
                perform(g, dry_run);
            }
        })
        .await?;

    println!("Done.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(engine: &mut GestureEngine, t0: Instant, ms: u64, s: AcmSample) -> Option<Gesture> {
        engine.on_sample(s, t0 + Duration::from_millis(ms))
    }

    /// 50 Hz worth of still samples covering calibration, gravity along +z.
    fn calibrate(engine: &mut GestureEngine, t0: Instant) -> u64 {
        let mut ms = 0;
        while !engine.calibrated() {
            assert!(feed(engine, t0, ms, AcmSample { x: 0, y: 0, z: 1000 }).is_none());
            ms += 20;
            assert!(ms < 10_000, "calibration never completed");
        }
        ms
    }

    #[test]
    fn tilt_fires_after_dwell_and_rearms_on_release() {
        let t0 = Instant::now();
        let mut e = GestureEngine::new(25.0, 0.6, t0);
        let mut ms = calibrate(&mut e, t0);

        // Roll ~45° (y shares gravity with z): must dwell before firing.
        let tilted = AcmSample { x: 0, y: 700, z: 700 };
        let mut fired = None;
        for _ in 0..100 {
            fired = feed(&mut e, t0, ms, tilted);
            ms += 20;
            if fired.is_some() {
                break;
            }
        }
        assert_eq!(fired, Some(Gesture::TiltRight));

        // Held past the threshold: must NOT re-fire until released.
        for _ in 0..100 {
            assert_eq!(feed(&mut e, t0, ms, tilted), None);
            ms += 20;
        }

        // Return to neutral, then tilt again → fires again.
        for _ in 0..50 {
            assert_eq!(feed(&mut e, t0, ms, AcmSample { x: 0, y: 0, z: 1000 }), None);
            ms += 20;
        }
        let mut fired = None;
        for _ in 0..100 {
            fired = feed(&mut e, t0, ms, tilted);
            ms += 20;
            if fired.is_some() {
                break;
            }
        }
        assert_eq!(fired, Some(Gesture::TiltRight));
    }

    #[test]
    fn double_shake_detected() {
        let t0 = Instant::now();
        let mut e = GestureEngine::new(25.0, 0.6, t0);
        let mut ms = calibrate(&mut e, t0);

        let still = AcmSample { x: 0, y: 0, z: 1000 };
        let spike = AcmSample { x: 1800, y: 0, z: 1000 };

        // Two spikes 200 ms apart, then settle.
        assert_eq!(feed(&mut e, t0, ms, spike), None);
        ms += 200;
        assert_eq!(feed(&mut e, t0, ms, spike), None);
        let mut fired = None;
        for _ in 0..50 {
            ms += 20;
            fired = feed(&mut e, t0, ms, still);
            if fired.is_some() {
                break;
            }
        }
        assert_eq!(fired, Some(Gesture::DoubleShake));
    }

    #[test]
    fn shake_does_not_fire_tilt() {
        let t0 = Instant::now();
        let mut e = GestureEngine::new(25.0, 0.6, t0);
        let mut ms = calibrate(&mut e, t0);

        // Violent alternating shake that also swings the tilt angles around:
        // only a shake gesture may come out, never a tilt.
        for i in 0..40 {
            let s = if i % 2 == 0 {
                AcmSample { x: 1800, y: 1500, z: -400 }
            } else {
                AcmSample { x: -1800, y: -1500, z: 1000 }
            };
            let fired = feed(&mut e, t0, ms, s);
            assert!(
                !matches!(
                    fired,
                    Some(Gesture::TiltLeft | Gesture::TiltRight | Gesture::TiltUp | Gesture::TiltDown)
                ),
                "tilt fired during shake: {fired:?}"
            );
            ms += 130;
        }
    }
}
