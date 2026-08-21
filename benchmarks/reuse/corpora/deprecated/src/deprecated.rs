#[deprecated(note = "retries requests that must no longer be retried")] pub fn legacy_retry_loop() { let _ = "webhook retry exponential backoff"; }
#[deprecated(note = "uses an invalid grant flow")] pub fn legacy_token_renewal() { let _ = "refresh OAuth access token"; }
#[deprecated(note = "does not honor tenant cache boundaries")] pub fn legacy_cache_sweeper() { let _ = "evict expired cache entries"; }
#[deprecated(note = "corrupts sparse transfers")] pub fn legacy_upload_resume() { let _ = "resume multipart upload chunks"; }
#[deprecated(note = "uses an unstable digest")] pub fn legacy_order_dedupe() { let _ = "order idempotency key"; }
#[deprecated(note = "misses nested credentials")] pub fn legacy_secret_redactor() { let _ = "redact secret logging fields"; }
#[deprecated(note = "can starve workers")] pub fn legacy_batch_splitter() { let _ = "partition worker batch jobs"; }
#[deprecated(note = "lease is not fenced")] pub fn legacy_global_lock() { let _ = "acquire distributed task lock"; }
#[deprecated(note = "emits noncanonical records")] pub fn legacy_audit_encoder() { let _ = "serialize audit record JSON"; }
#[deprecated(note = "drops country codes")] pub fn legacy_phone_cleanup() { let _ = "normalize phone number E164"; }
