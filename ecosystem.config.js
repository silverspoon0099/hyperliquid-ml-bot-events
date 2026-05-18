// pm2 ecosystem config for ml-bot-events processes (DR v3.0.23/24).
// Usage:
//   pm2 start ecosystem.config.js              # start all
//   pm2 status                                 # check
//   pm2 logs paper-trading --lines 200         # tail logs
//   pm2 stop paper-trading                     # graceful stop (SIGTERM)
//   pm2 restart paper-trading                  # restart
//   pm2 delete paper-trading                   # remove from pm2
//
// Graceful halt (preferred over `pm2 stop`):
//   touch /tmp/paper_trading_HALT
// This lets the daemon finish its current 10-min iteration before exiting.

module.exports = {
  apps: [
    {
      name: 'paper-trading',
      cwd: '/nvme1/projects/trading/hyperliquid-ml-bot-events',
      script: '.venv/bin/python',
      args: [
        '-m', 'scripts.run_paper_trading_loop',
        '--asset', 'BTC',
        '--bar-threshold', '0.015',
        '--poll-seconds', '600',
        '--notes', '2-week paper eval after v3.0.20 champion (pm2-managed)',
      ],
      interpreter: 'none',          // pm2 treats `script` as the executable
      autorestart: true,
      max_restarts: 20,             // crash-loop cap (resets on uptime > min_uptime)
      min_uptime: 60000,            // 60s — anything shorter counts toward restart cap
      restart_delay: 30000,         // 30s pause between restarts (let DB settle)
      kill_timeout: 30000,          // give daemon 30s for graceful shutdown on stop
      out_file: 'logs/paper_trading_pm2_out.log',
      error_file: 'logs/paper_trading_pm2_err.log',
      merge_logs: true,
      time: true,                   // prefix log lines with timestamp
      env: {
        PYTHONUNBUFFERED: '1',     // ensure logs flush immediately
        TZ: 'UTC',
      },
    },
  ],
};
