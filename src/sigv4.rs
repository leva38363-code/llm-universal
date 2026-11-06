//! AWS Signature Version 4 for a request without a body or with a known body hash.
//! Used for Bedrock. Checked against the official `get-vanilla` test vector.

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hmac(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

/// Everything needed to sign one request.
pub struct Signing<'a> {
    pub method: &'a str,
    /// The raw request path, not percent-encoded: it is encoded here as AWS requires.
    pub path: &'a str,
    pub query: &'a str,
    /// Must include `host` and `x-amz-date`; names are lowercased and sorted here.
    pub headers: &'a [(&'a str, &'a str)],
    pub payload_sha256_hex: &'a str,
    pub access_key: &'a str,
    pub secret_key: &'a str,
    pub region: &'a str,
    pub service: &'a str,
    /// Full `YYYYMMDDTHHMMSSZ` timestamp; its first 8 characters are the date.
    pub amz_date: &'a str,
}

#[allow(dead_code)] // подключается к провайдеру Bedrock отдельным шагом
/// Returns the `Authorization` header value, or `None` if the input is malformed
/// (`amz_date` not `YYYYMMDDTHHMMSSZ`, or non-ASCII / control characters in headers).
pub fn authorization(input: &Signing<'_>) -> Option<String> {
    let Signing {
        method,
        path,
        query,
        headers,
        payload_sha256_hex,
        access_key,
        secret_key,
        region,
        service,
        amz_date,
    } = *input;
    let well_formed = amz_date.len() == 16
        && amz_date.is_ascii()
        && amz_date.as_bytes()[8] == b'T'
        && amz_date.ends_with('Z');
    let clean = |t: &str| t.is_ascii() && !t.chars().any(|c| c.is_control());
    let digits = |t: &str| !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit());
    let token = |t: &str| !t.is_empty() && t.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-');
    // Access key ids are alphanumeric; anything else (`,`, `/`, spaces) would change the
    // Authorization header's structure.
    let key_ok = !access_key.is_empty()
        && access_key.len() <= 128
        && access_key.bytes().all(|b| b.is_ascii_alphanumeric());
    if !well_formed
        || !digits(&amz_date[..8])
        || !digits(&amz_date[9..15])
        || !key_ok
        || !token(region)
        || !token(service)
        || !headers.iter().all(|(k, v)| clean(k) && clean(v))
    {
        return None;
    }
    let date = &amz_date[..8];
    let mut sorted: Vec<(String, String)> = headers
        .iter()
        .map(|(k, v)| (k.to_ascii_lowercase(), collapse_spaces(v)))
        .collect();
    sorted.sort();
    let canonical_headers: String = sorted.iter().map(|(k, v)| format!("{k}:{v}\n")).collect();
    let signed_headers = sorted
        .iter()
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>()
        .join(";");

    let canonical_query = canonical_query(query);
    let canonical_request = format!(
        "{method}\n{}\n{canonical_query}\n{canonical_headers}\n{signed_headers}\n{payload_sha256_hex}",
        canonical_uri(path, service)
    );
    let scope = format!("{date}/{region}/{service}/aws4_request");
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        hex(&Sha256::digest(canonical_request.as_bytes()))
    );

    let k_date = hmac(format!("AWS4{secret_key}").as_bytes(), date.as_bytes());
    let k_region = hmac(&k_date, region.as_bytes());
    let k_service = hmac(&k_region, service.as_bytes());
    let k_signing = hmac(&k_service, b"aws4_request");
    let signature = hex(&hmac(&k_signing, string_to_sign.as_bytes()));

    Some(format!(
        "AWS4-HMAC-SHA256 Credential={access_key}/{scope}, SignedHeaders={signed_headers}, Signature={signature}"
    ))
}

/// AWS: each path segment is percent-encoded (unreserved characters `A-Z a-z 0-9 - _ . ~` kept);
/// every service except S3 encodes it a second time, so `:` in a Bedrock model id becomes `%253A`.
fn canonical_uri(path: &str, service: &str) -> String {
    if path.is_empty() {
        return "/".to_string();
    }
    let encode = |s: &str| -> String {
        s.bytes()
            .map(|b| {
                if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
                    (b as char).to_string()
                } else {
                    format!("%{b:02X}")
                }
            })
            .collect()
    };
    path.split('/')
        .map(|segment| {
            let once = encode(segment);
            if service == "s3" {
                once
            } else {
                encode(&once)
            }
        })
        .collect::<Vec<_>>()
        .join("/")
}

/// AWS: trim, and replace runs of spaces inside a header value with one space.
fn collapse_spaces(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// AWS: query parameters sorted by name, then by value; `key=value` pairs kept as given
/// (the caller must already percent-encode them as AWS requires).
fn canonical_query(query: &str) -> String {
    if query.is_empty() {
        return String::new();
    }
    let mut pairs: Vec<(&str, &str)> = query
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| p.split_once('=').unwrap_or((p, "")))
        .collect();
    pairs.sort();
    pairs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("&")
}

#[cfg(test)]
mod tests {
    use super::*;

    // Официальный тест AWS (get-vanilla): ключи из документации AWS, ожидаемая подпись из набора тестов.
    #[test]
    fn matches_aws_get_vanilla_vector() {
        let empty_sha = hex(&Sha256::digest(b""));
        let auth = authorization(&Signing {
            method: "GET",
            path: "/",
            query: "",
            headers: &[
                ("Host", "example.amazonaws.com"),
                ("X-Amz-Date", "20150830T123600Z"),
            ],
            payload_sha256_hex: &empty_sha,
            access_key: "AKIDEXAMPLE",
            secret_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            region: "us-east-1",
            service: "service",
            amz_date: "20150830T123600Z",
        });
        assert_eq!(
            auth.as_deref(),
            Some("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, SignedHeaders=host;x-amz-date, Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31")
        );
    }

    #[test]
    fn canonical_uri_encodes_segments_and_double_encodes_outside_s3() {
        assert_eq!(canonical_uri("/", "bedrock"), "/");
        assert_eq!(canonical_uri("/model/anthropic.claude-v2:1/invoke", "bedrock"), "/model/anthropic.claude-v2%253A1/invoke");
        assert_eq!(canonical_uri("/bucket/a b", "s3"), "/bucket/a%20b");
    }

    #[test]
    fn unsafe_access_key_or_region_is_refused() {
        let empty_sha = hex(&Sha256::digest(b""));
        for (key, region) in [("AK,ID", "us-east-1"), ("AK/ID", "us-east-1"), ("AKID", "us east"), ("", "us-east-1")] {
            let r = authorization(&Signing {
                method: "GET",
                path: "/",
                query: "",
                headers: &[("Host", "example.amazonaws.com")],
                payload_sha256_hex: &empty_sha,
                access_key: key,
                secret_key: "S",
                region,
                service: "s",
                amz_date: "20150830T123600Z",
            });
            assert!(r.is_none(), "{key:?} {region:?}");
        }
    }

    #[test]
    fn malformed_input_returns_none_instead_of_panicking() {
        let empty_sha = hex(&Sha256::digest(b""));
        for bad_date in ["", "2015", "2015083", "€€€", "20150830X123600Z"] {
            let r = authorization(&Signing {
                method: "GET",
                path: "/",
                query: "",
                headers: &[("Host", "example.amazonaws.com")],
                payload_sha256_hex: &empty_sha,
                access_key: "A",
                secret_key: "S",
                region: "us-east-1",
                service: "s",
                amz_date: bad_date,
            });
            assert!(r.is_none(), "{bad_date:?}");
        }
        let r = authorization(&Signing {
            method: "GET",
            path: "/",
            query: "",
            headers: &[("Host", "example.amazonaws.com"), ("x-Ünï", "v\r\nx")],
            payload_sha256_hex: &empty_sha,
            access_key: "A",
            secret_key: "S",
            region: "us-east-1",
            service: "s",
            amz_date: "20150830T123600Z",
        });
        assert!(r.is_none());
    }

    #[test]
    fn query_is_sorted_and_header_spaces_collapsed() {
        assert_eq!(canonical_query("b=2&a=1&a=0"), "a=0&a=1&b=2");
        assert_eq!(canonical_query(""), "");
        assert_eq!(collapse_spaces("  a   b\tc  "), "a b c");
    }
}
