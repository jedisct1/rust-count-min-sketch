//! Count-min sketch with conservative updates.
//!
//! Four variants are provided, differing only in the width of their
//! counters: [`CountMinSketch8`], [`CountMinSketch16`], [`CountMinSketch32`]
//! and [`CountMinSketch64`].

use std::borrow::Borrow;
use std::cmp::max;
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem;

use siphasher::sip128::{Hasher128, SipHasher13};

macro_rules! cms_define {
    ($CountMinSketch:ident, $Counter:ty) => {
        pub struct $CountMinSketch<K> {
            counters: Vec<$Counter>,
            sip_key: [u8; 16],
            mask: usize,
            k_num: usize,
            reset_idx: usize,
            phantom_k: PhantomData<K>,
        }

        impl<K> $CountMinSketch<K>
        where
            K: Hash,
        {
            pub fn new(
                capacity: usize,
                probability: f64,
                tolerance: f64,
            ) -> Result<Self, &'static str> {
                let width = Self::optimal_width(capacity, tolerance);
                let k_num = Self::optimal_k_num(probability);
                Ok(Self {
                    counters: vec![0; width * k_num],
                    sip_key: Self::random_key(),
                    mask: width - 1,
                    k_num,
                    reset_idx: 0,
                    phantom_k: PhantomData,
                })
            }

            pub fn add<Q>(&mut self, key: &Q, value: $Counter)
            where
                Q: Hash + ?Sized,
                K: Borrow<Q>,
            {
                let (h1, h2) = self.hash(key);
                let lowest = self.lowest(h1, h2);
                let updated = lowest.saturating_add(value);
                for k_i in 0..self.k_num {
                    let offset = self.offset(h1, h2, k_i);
                    let counter = &mut self.counters[offset];
                    if *counter == lowest {
                        *counter = updated;
                    }
                }
            }

            pub fn increment<Q>(&mut self, key: &Q)
            where
                Q: Hash + ?Sized,
                K: Borrow<Q>,
            {
                self.add(key, 1)
            }

            pub fn estimate<Q>(&self, key: &Q) -> $Counter
            where
                Q: Hash + ?Sized,
                K: Borrow<Q>,
            {
                let (h1, h2) = self.hash(key);
                self.lowest(h1, h2)
            }

            pub fn estimate_memory(
                capacity: usize,
                probability: f64,
                tolerance: f64,
            ) -> Result<usize, &'static str> {
                let width = Self::optimal_width(capacity, tolerance);
                let k_num = Self::optimal_k_num(probability);
                Ok(width * mem::size_of::<$Counter>() * k_num)
            }

            pub fn clear(&mut self) {
                self.counters.fill(0);
                self.reset_idx = 0;
                self.sip_key = Self::random_key();
            }

            pub fn reset(&mut self) {
                for counter in &mut self.counters {
                    *counter /= 2;
                }
                self.reset_idx = 0;
            }

            pub fn reset_next(&mut self) -> Option<usize> {
                let idx = self.reset_idx;
                for k_i in 0..self.k_num {
                    let offset = self.slot(k_i, idx);
                    self.counters[offset] /= 2;
                }
                let next = (idx + 1) & self.mask;
                self.reset_idx = next;
                if next != 0 {
                    Some(next)
                } else {
                    None
                }
            }

            fn optimal_width(capacity: usize, tolerance: f64) -> usize {
                let e = tolerance / (capacity as f64);
                let width = (2.0 / e).round() as usize;
                max(2, width)
                    .checked_next_power_of_two()
                    .expect("Width would be way too large")
            }

            fn optimal_k_num(probability: f64) -> usize {
                max(1, ((1.0 - probability).ln() / 0.5f64.ln()) as usize)
            }

            fn random_key() -> [u8; 16] {
                let mut key = [0u8; 16];
                getrandom::fill(&mut key).expect("random source unavailable");
                key
            }

            fn hash<Q>(&self, key: &Q) -> (u64, u64)
            where
                Q: Hash + ?Sized,
            {
                let mut sip = SipHasher13::new_with_key(&self.sip_key);
                key.hash(&mut sip);
                sip.finish128().as_u64()
            }

            fn lowest(&self, h1: u64, h2: u64) -> $Counter {
                let mut lowest = <$Counter>::MAX;
                for k_i in 0..self.k_num {
                    lowest = lowest.min(self.counters[self.offset(h1, h2, k_i)]);
                }
                lowest
            }

            #[inline]
            fn offset(&self, h1: u64, h2: u64, k_i: usize) -> usize {
                let column = h1.wrapping_add((k_i as u64).wrapping_mul(h2)) as usize & self.mask;
                self.slot(k_i, column)
            }

            #[inline]
            fn slot(&self, k_i: usize, column: usize) -> usize {
                k_i * (self.mask + 1) + column
            }
        }

        // A derive would add a `K: Clone` bound through PhantomData<K>.
        impl<K> Clone for $CountMinSketch<K> {
            fn clone(&self) -> Self {
                Self {
                    counters: self.counters.clone(),
                    ..*self
                }
            }
        }
    };
}

cms_define!(CountMinSketch8, u8);
cms_define!(CountMinSketch16, u16);
cms_define!(CountMinSketch32, u32);
cms_define!(CountMinSketch64, u64);

#[cfg(test)]
mod tests {
    use crate::{CountMinSketch16, CountMinSketch32, CountMinSketch64, CountMinSketch8};

    #[test]
    fn test_overflow() {
        let mut cms = CountMinSketch8::<&str>::new(100, 0.95, 10.0).unwrap();
        for _ in 0..300 {
            cms.increment("key");
        }
        assert_eq!(cms.estimate("key"), u8::MAX);
    }

    #[test]
    fn test_increment() {
        let mut cms = CountMinSketch16::<&str>::new(100, 0.95, 10.0).unwrap();
        for _ in 0..300 {
            cms.increment("key");
        }
        assert_eq!(cms.estimate("key"), 300);
    }

    #[test]
    fn test_increment_multi() {
        let mut cms = CountMinSketch64::<u64>::new(100, 0.99, 2.0).unwrap();
        for i in 0..1_000_000 {
            cms.increment(&(i % 100));
        }
        for key in 0..100 {
            assert!(cms.estimate(&key) >= 9_000);
        }
        cms.reset();
        for key in 0..100 {
            assert!(cms.estimate(&key) < 11_000);
        }
    }

    #[test]
    fn test_clone_and_clear() {
        let mut cms = CountMinSketch32::<String>::new(100, 0.95, 10.0).unwrap();
        for _ in 0..9 {
            cms.increment("key");
        }
        cms.increment(&"key".to_string());
        let snapshot = cms.clone();
        cms.increment("key");
        assert_eq!(snapshot.estimate("key"), 10);
        assert_eq!(cms.estimate("key"), 11);
        cms.clear();
        assert_eq!(cms.estimate("key"), 0);
        assert_eq!(snapshot.estimate("key"), 10);
    }

    #[test]
    fn test_reset_next_full_cycle() {
        let mut cms = CountMinSketch32::<&str>::new(100, 0.95, 10.0).unwrap();
        for _ in 0..100 {
            cms.increment("key");
        }
        while cms.reset_next().is_some() {}
        assert_eq!(cms.estimate("key"), 50);
    }
}
