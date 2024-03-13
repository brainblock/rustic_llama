use std::{
    io::Read,
    time::{SystemTime, SystemTimeError, UNIX_EPOCH},
};

use anyhow::Result;
use bytemuck::{from_bytes, Pod};
use std::mem::{self};

pub fn time_in_ms() -> Result<u128, SystemTimeError> {
    let current_system_time = SystemTime::now();
    let duration_since_epoch = current_system_time.duration_since(UNIX_EPOCH)?;
    let milliseconds_timestamp = duration_since_epoch.as_millis();

    Ok(milliseconds_timestamp)
}

pub fn read_value<T: Pod>(reader: &mut impl Read) -> Result<T> {
    let size = mem::size_of::<T>();
    let mut bytes: Vec<u8> = vec![0u8; size];
    reader.read(bytes.as_mut_slice())?;
    let val = from_bytes(&bytes);
    Ok(*val)
}
