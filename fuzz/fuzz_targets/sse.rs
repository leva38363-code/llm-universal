#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let split = data.first().map(|b| usize::from(*b)).unwrap_or(0);
    llm_universal::fuzz_sse(data.get(1..).unwrap_or(&[]), split);
});
