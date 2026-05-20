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
        '--confidence-threshold', '0.58',   // v3.0.20 champion (config default is 0.60 spec freeze)
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
    {
      // DR v3.0.25 — Streamlit trade tracking dashboard
      // Fronted by nginx at: http://<host>/dashboard/
      // Direct (localhost only): http://127.0.0.1:8501
      name: 'dashboard',
      cwd: '/nvme1/projects/trading/hyperliquid-ml-bot-events',
      script: '.venv/bin/streamlit',
      args: [
        'run', 'dashboard/app.py',
        '--server.port', '8501',
        '--server.address', '127.0.0.1',         // localhost only — nginx fronts public access
        '--server.headless', 'true',             // no auto-open browser
        '--server.baseUrlPath', 'dashboard',     // for nginx subpath /dashboard/
        '--server.enableCORS', 'false',          // nginx handles
        '--server.enableXsrfProtection', 'false', // nginx handles
        '--browser.gatherUsageStats', 'false',   // privacy
      ],
      interpreter: 'none',
      autorestart: true,
      max_restarts: 10,
      min_uptime: 30000,
      restart_delay: 5000,
      out_file: 'logs/dashboard_pm2_out.log',
      error_file: 'logs/dashboard_pm2_err.log',
      merge_logs: true,
      time: true,
      env: {
        PYTHONUNBUFFERED: '1',
        TZ: 'UTC',
      },
    },
  ],
};
