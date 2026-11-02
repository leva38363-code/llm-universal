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
    if !well_formed || !headers.iter().all(|(k, v)| clean(k) && clean(v)) {
        return None;
    }
    let date = &amz_date[..8];
    let mut sorted: Vec<(String, String)> = headers
        .iter()
        .map(|(k, v)| (k.to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    sorted.sort();
    let canonical_headers: String = sorted.iter().map(|(k, v)| format!("{k}:{v}\n")).collect();
    let signed_headers = sorted.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>().join(";");

    let canonical_request = format!(
        "{method}\n{path}\n{query}\n{canonical_headers}\n{signed_headers}\n{payload_sha256_hex}"
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
            headers: &[("Host", "example.amazonaws.com"), ("X-Amz-Date", "20150830T123600Z")],
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
    fn malformed_input_returns_none_instead_of_panicking() {
        let empty_sha = hex(&Sha256::digest(b""));
        for bad_date in ["", "2015", "2015083", "€€€", "20150830X123600Z"] {
            let r = authorization(&Signing {
                method: "GET", path: "/", query: "",
                headers: &[("Host", "example.amazonaws.com")],
                payload_sha256_hex: &empty_sha, access_key: "A", secret_key: "S",
                region: "us-east-1", service: "s", amz_date: bad_date,
            });
            assert!(r.is_none(), "{bad_date:?}");
        }
        let r = authorization(&Signing {
            method: "GET", path: "/", query: "",
            headers: &[("Host", "example.amazonaws.com"), ("x-Ünï", "v\r\nx")],
            payload_sha256_hex: &empty_sha, access_key: "A", secret_key: "S",
            region: "us-east-1", service: "s", amz_date: "20150830T123600Z",
        });
        assert!(r.is_none());
    }
}
