# Whatsapp AI bot using Gemini
Basically the bot communicates with Gemini's API using a webhook through messages you send to the Whatsapp bot.
Since its a WIP, the memory is only handled locally, so once you restart the AI forgets your previous messages.
You can get a free Gemini API key in Google's AI studio and you'll be able to choose between more models even the newer 3.0, you'll have to change it in the code tho.
For the Whatsapp bot part you'll have to create an account in Meta developers and create a whatsapp app, get the API key and phone number ID and setup a password for the webhook.
Put all that info in the .env and setup ngrok or another tunneling service to the http port 5000 and use it with /webhook at the end to connect it to the bot.
Make sure you also subcribe to messages on the webhook fields.
