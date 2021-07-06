use log::LevelFilter;

use logger::Logger;
static LOGGER: Logger = Logger;

#[profiling::function]
fn main() {
    log::set_logger(&LOGGER)
        .map(|()| log::set_max_level(LevelFilter::Trace))
        .unwrap();
    log::info!("Hello, world!");
}

mod logger {
    use log::{Level, Log, Metadata, Record};
    use sdl2::messagebox::{show_simple_message_box, MessageBoxFlag};
    use std::thread;

    pub struct Logger;

    impl Log for Logger {
        fn enabled(&self, _metadata: &Metadata) -> bool {
            true
        }

        fn log(&self, record: &Record) {
            if self.enabled(record.metadata()) {
                let message = format!("{}", record.args());
                let (color_code, color_end) = if cfg!(target_family = "unix") {
                    let start = match record.level() {
                        Level::Trace => "\u{1B}[34m", /* blue */
                        Level::Debug => "\u{1B}[36m", /* cyan */
                        Level::Info => "\u{1B}[32m",  /* green */
                        Level::Warn => "\u{1B}[33m",  /* yellow */
                        Level::Error => "\u{1B}[31m", /* red */
                    };
                    (start, "\u{1B}[m")
                } else {
                    ("", "")
                };
                eprintln!(
                    "{}[{}:{}] {}{}",
                    color_code,
                    record.file().unwrap_or(""),
                    record.line().unwrap_or(0),
                    message,
                    color_end,
                );
                if record.level() == Level::Error {
                    thread::spawn(move || {
                        if let Some(title_len) = message.char_indices().position(|(_, c)| c == ':')
                        {
                            let title = &message[..title_len];
                            let message = &message[(title_len + 2).min(message.len())..];
                            let _ = show_simple_message_box(
                                MessageBoxFlag::ERROR,
                                title,
                                message,
                                None,
                            );
                        } else {
                            let _ = show_simple_message_box(
                                MessageBoxFlag::ERROR,
                                "Error",
                                &message,
                                None,
                            );
                        }
                    });
                }
            }
        }

        fn flush(&self) {}
    }
}
