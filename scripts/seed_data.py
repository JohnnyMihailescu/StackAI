#!/usr/bin/env python3
"""Seed the StackAI database with test data.

Usage:
    python scripts/seed_data.py --library recipes
    python scripts/seed_data.py --library support
    python scripts/seed_data.py --library products
    python scripts/seed_data.py --library all
"""

import argparse
import httpx

BASE_URL = "http://localhost:8000/api/v1"

# Each library contains documents, each document contains sequential sentence chunks
LIBRARIES = {
    "recipes": {
        "name": "Recipe Collection",
        "documents": {
            "spaghetti_carbonara": {
                "name": "Spaghetti Carbonara",
                "chunks": [
                    "Spaghetti Carbonara is a classic Roman pasta dish made with eggs, cheese, and cured pork.",
                    "You'll need 400g spaghetti, 200g guanciale, 4 egg yolks, 100g pecorino romano, and black pepper.",
                    "Bring a large pot of salted water to boil and cook the spaghetti until al dente.",
                    "Cut the guanciale into small strips and crisp it in a dry pan over medium heat.",
                    "Whisk together the egg yolks, grated pecorino, and plenty of black pepper in a bowl.",
                    "Reserve a cup of pasta water before draining the cooked spaghetti.",
                    "Toss the hot pasta with the guanciale and rendered fat, then remove from heat.",
                    "Stir in the egg mixture quickly, adding pasta water as needed to create a creamy sauce.",
                    "Never add the eggs over direct heat or they will scramble.",
                    "Serve immediately with extra pecorino and black pepper.",
                ]
            },
            "thai_green_curry": {
                "name": "Thai Green Curry",
                "chunks": [
                    "Thai Green Curry is a fragrant coconut-based curry from central Thailand.",
                    "Gather 400ml coconut milk, 2 tbsp green curry paste, 500g chicken thighs, Thai basil, and fish sauce.",
                    "Cut the chicken into bite-sized pieces and slice any vegetables you're using.",
                    "Heat a wok over high heat and add a splash of the thick coconut cream from the top of the can.",
                    "Fry the curry paste in the coconut cream for 2 minutes until fragrant.",
                    "Add the chicken pieces and stir-fry until they start to turn white on the outside.",
                    "Pour in the remaining coconut milk and bring to a gentle simmer.",
                    "Add vegetables like bamboo shoots, Thai eggplant, or bell peppers.",
                    "Season with fish sauce and a little palm sugar to balance the flavors.",
                    "Finish with fresh Thai basil leaves and serve over jasmine rice.",
                ]
            },
            "chocolate_chip_cookies": {
                "name": "Chocolate Chip Cookies",
                "chunks": [
                    "These chocolate chip cookies are crispy on the edges and chewy in the center.",
                    "You'll need 225g butter, 200g brown sugar, 100g white sugar, 2 eggs, and 300g flour.",
                    "Don't forget 1 tsp baking soda, 1 tsp salt, 2 tsp vanilla, and 350g chocolate chips.",
                    "Cream the softened butter with both sugars until light and fluffy, about 3 minutes.",
                    "Beat in the eggs one at a time, then add the vanilla extract.",
                    "Whisk together the flour, baking soda, and salt in a separate bowl.",
                    "Gradually mix the dry ingredients into the wet, being careful not to overmix.",
                    "Fold in the chocolate chips and chill the dough for at least 30 minutes.",
                    "Scoop rounded tablespoons onto a baking sheet lined with parchment paper.",
                    "Bake at 375°F for 9-11 minutes until the edges are golden but centers look slightly underdone.",
                ]
            },
            "chicken_tikka_masala": {
                "name": "Chicken Tikka Masala",
                "chunks": [
                    "Chicken Tikka Masala features tender marinated chicken in a creamy tomato-based sauce.",
                    "For the marinade, combine 500g chicken breast with yogurt, garam masala, turmeric, and lemon juice.",
                    "Let the chicken marinate for at least 2 hours, or overnight for best results.",
                    "Thread the chicken onto skewers and grill or broil until slightly charred.",
                    "For the sauce, sauté diced onions in butter until soft and golden.",
                    "Add minced garlic, ginger, and spices like cumin, coriander, and paprika.",
                    "Pour in crushed tomatoes and simmer for 15 minutes until the sauce thickens.",
                    "Stir in heavy cream and the grilled chicken pieces.",
                    "Simmer together for another 10 minutes to let the flavors meld.",
                    "Garnish with fresh cilantro and serve with warm naan bread and basmati rice.",
                ]
            },
            "french_onion_soup": {
                "name": "French Onion Soup",
                "chunks": [
                    "French Onion Soup is a rich, savory soup topped with crusty bread and melted cheese.",
                    "You'll need 4 large onions, 4 tbsp butter, beef broth, dry white wine, and gruyère cheese.",
                    "Slice the onions thinly and cook them in butter over medium-low heat.",
                    "Caramelizing onions properly takes 45 minutes to an hour - don't rush this step.",
                    "Stir occasionally and add a pinch of salt to help draw out moisture.",
                    "Once the onions are deep golden brown, add a splash of wine to deglaze.",
                    "Pour in the beef broth and add a bay leaf and fresh thyme.",
                    "Simmer for 20 minutes to develop the flavors.",
                    "Ladle into oven-safe bowls and top with a slice of crusty bread.",
                    "Cover generously with grated gruyère and broil until bubbly and golden.",
                ]
            },
        }
    },
    "support": {
        "name": "Support Knowledge Base",
        "documents": {
            "account_login": {
                "name": "Account & Login",
                "chunks": [
                    "To reset your password, click 'Forgot Password' on the login page.",
                    "You'll receive an email with a reset link that expires in 24 hours.",
                    "If you don't see the email, check your spam folder or try again.",
                    "To enable two-factor authentication, go to Settings and select Security.",
                    "We support authenticator apps like Google Authenticator or Authy.",
                    "If you lose access to your 2FA device, use one of your backup codes to log in.",
                    "Backup codes are shown once when you enable 2FA - store them somewhere safe.",
                    "To change your email address, verify your identity with your current password first.",
                ]
            },
            "billing_payments": {
                "name": "Billing & Payments",
                "chunks": [
                    "We accept all major credit cards including Visa, Mastercard, and American Express.",
                    "Charges appear on your statement as 'STACKAI SERVICES'.",
                    "To request a refund, contact support within 30 days of your purchase.",
                    "Refunds are processed within 5-7 business days to your original payment method.",
                    "You can download invoices from the Billing section of your account settings.",
                    "To update your payment method, go to Settings and select Billing.",
                    "If your payment fails, we'll retry automatically and notify you by email.",
                    "Annual subscriptions receive a 20% discount compared to monthly billing.",
                ]
            },
            "getting_started": {
                "name": "Getting Started",
                "chunks": [
                    "Welcome to StackAI - let's get you set up in just a few minutes.",
                    "First, create your account using your email or sign in with Google.",
                    "After signing in, you'll be prompted to create your first project.",
                    "Each project can contain multiple libraries for organizing your data.",
                    "Upload your documents using the web interface or our REST API.",
                    "Documents are automatically split into chunks and embedded for search.",
                    "Try your first search by entering a natural language query.",
                    "Check out our API documentation for programmatic access to all features.",
                ]
            },
            "troubleshooting": {
                "name": "Troubleshooting",
                "chunks": [
                    "If the application won't load, try clearing your browser cache and cookies.",
                    "Make sure you're using a supported browser: Chrome, Firefox, Safari, or Edge.",
                    "For slow search results, check your internet connection speed.",
                    "API timeout errors usually indicate the request payload is too large.",
                    "Try reducing batch sizes if you're uploading many documents at once.",
                    "If embeddings fail to generate, verify your API key is valid and has quota remaining.",
                    "Check our status page at status.stackai.com for any ongoing incidents.",
                    "For persistent issues, contact support with your request ID from the error message.",
                ]
            },
            "data_privacy": {
                "name": "Data & Privacy",
                "chunks": [
                    "Your data is encrypted at rest using AES-256 encryption.",
                    "All connections use TLS 1.3 to protect data in transit.",
                    "We never share your data with third parties without your explicit consent.",
                    "To export your data, go to Settings and select 'Download My Data'.",
                    "Data exports are provided in JSON format and include all your documents and metadata.",
                    "To delete your account, go to Settings and select 'Delete Account'.",
                    "Account deletion is permanent and removes all your data within 30 days.",
                    "We retain anonymized usage analytics but never your actual content.",
                ]
            },
        }
    },
    "products": {
        "name": "Product Manuals",
        "documents": {
            "wireless_headphones": {
                "name": "Wireless Headphones WH-1000",
                "chunks": [
                    "The WH-1000 wireless headphones feature active noise cancellation and 30-hour battery life.",
                    "In the box you'll find the headphones, USB-C charging cable, 3.5mm audio cable, and carrying case.",
                    "To charge, connect the USB-C cable to the port on the right ear cup.",
                    "A full charge takes 3 hours and the LED turns green when complete.",
                    "Press and hold the power button for 3 seconds to turn on the headphones.",
                    "To pair with Bluetooth, hold the power button for 7 seconds until the LED flashes blue.",
                    "The headphones will appear as 'WH-1000' in your device's Bluetooth settings.",
                    "Toggle noise cancellation by pressing the ANC button on the left ear cup.",
                    "If audio cuts out, ensure you're within 10 meters of the connected device.",
                    "To reset the headphones, hold both the power and ANC buttons for 10 seconds.",
                ]
            },
            "smart_thermostat": {
                "name": "Smart Thermostat ST-200",
                "chunks": [
                    "The ST-200 Smart Thermostat learns your schedule and can reduce energy bills by up to 23%.",
                    "Before installation, turn off power to your HVAC system at the circuit breaker.",
                    "Remove your old thermostat and take a photo of the wire connections for reference.",
                    "Connect the labeled wires to the matching terminals on the ST-200 base plate.",
                    "Attach the base plate to the wall using the included screws and anchors.",
                    "Snap the display unit onto the base plate until it clicks into place.",
                    "Restore power and follow the on-screen setup wizard to connect to WiFi.",
                    "Download the companion app to control the thermostat remotely.",
                    "The thermostat learns your preferences over the first week of use.",
                    "If the display is unresponsive, check that all wires are securely connected.",
                ]
            },
            "espresso_machine": {
                "name": "Espresso Machine EM-500",
                "chunks": [
                    "The EM-500 features a 15-bar pump, built-in grinder, and automatic milk frother.",
                    "Before first use, fill the water tank and run two cycles without coffee to flush the system.",
                    "Fill the bean hopper with fresh espresso beans and select your grind size.",
                    "For espresso, use a fine grind setting between 2 and 4 on the dial.",
                    "Insert the portafilter and press the single or double shot button.",
                    "The machine will grind, tamp, and extract automatically in about 25 seconds.",
                    "To steam milk, place the steam wand just below the surface and turn the dial.",
                    "Keep the wand at an angle to create a spinning vortex for smooth microfoam.",
                    "Run water through the steam wand after each use to prevent milk buildup.",
                    "Descale the machine monthly using the cleaning cycle and descaling solution.",
                ]
            },
            "robot_vacuum": {
                "name": "Robot Vacuum RV-300",
                "chunks": [
                    "The RV-300 robot vacuum maps your home and provides up to 120 minutes of cleaning.",
                    "Place the charging dock against a wall with 1 meter clearance on each side.",
                    "Set the vacuum on the dock and allow it to charge fully before first use.",
                    "Press the clean button on top or use the app to start a cleaning session.",
                    "The vacuum will systematically cover your floors and return to dock when done.",
                    "Use the app to set no-go zones and schedule automatic cleaning times.",
                    "Empty the dustbin after each use by pressing the release button.",
                    "Clean the brush roll weekly to remove hair and debris buildup.",
                    "If the vacuum gets stuck, it will notify you through the app.",
                    "Replace the filter every 2 months and brush roll every 6 months for best performance.",
                ]
            },
            "mechanical_keyboard": {
                "name": "Mechanical Keyboard MK-75",
                "chunks": [
                    "The MK-75 mechanical keyboard features hot-swappable switches and RGB backlighting.",
                    "Connect the keyboard using the detachable USB-C cable or via Bluetooth.",
                    "To pair Bluetooth, press Fn+1, Fn+2, or Fn+3 to select a device slot.",
                    "Hold the combination for 3 seconds until the corresponding key flashes.",
                    "The keyboard can remember up to 3 Bluetooth devices and switch between them instantly.",
                    "Customize RGB lighting effects using Fn+Up/Down to cycle through modes.",
                    "Adjust brightness with Fn+Left/Right arrow keys.",
                    "To swap switches, use the included puller to remove keycaps then pull switches straight up.",
                    "The keyboard supports 3-pin and 5-pin mechanical switches.",
                    "If keys stop responding, try a different USB port or re-pair Bluetooth.",
                ]
            },
        }
    },
}


def create_library(client: httpx.Client, library_id: str, name: str) -> bool:
    """Create a library, return True if successful."""
    response = client.post("/libraries", json={
        "id": library_id,
        "name": name,
    })
    if response.status_code == 201:
        print(f"Created library: {library_id}")
        return True
    elif response.status_code == 409:
        print(f"Library already exists: {library_id}")
        return True
    else:
        print(f"Failed to create library: {response.text}")
        return False


def create_document(client: httpx.Client, library_id: str, doc_id: str, name: str) -> bool:
    """Create a document, return True if successful."""
    response = client.post(f"/libraries/{library_id}/documents", json={
        "id": doc_id,
        "library_id": library_id,
        "name": name,
    })
    if response.status_code == 201:
        print(f"  Created document: {name}")
        return True
    elif response.status_code == 409:
        print(f"  Document already exists: {doc_id}")
        return True
    else:
        print(f"  Failed to create document: {response.text}")
        return False


def create_chunks(client: httpx.Client, doc_id: str, texts: list[str]) -> int:
    """Create chunks for a document, return count created."""
    chunks = [
        {"id": f"{doc_id}_chunk_{i}", "document_id": doc_id, "text": text}
        for i, text in enumerate(texts, 1)
    ]
    response = client.post(f"/documents/{doc_id}/chunks/batch", json={"chunks": chunks})
    if response.status_code == 201:
        count = response.json()["created_count"]
        print(f"    Created {count} chunks")
        return count
    else:
        print(f"    Failed to create chunks: {response.text}")
        return 0


def seed_library(client: httpx.Client, library_key: str) -> int:
    """Seed a single library with data."""
    library_data = LIBRARIES[library_key]
    library_id = f"{library_key}_lib"

    print(f"\n{'='*50}")
    print(f"Seeding library: {library_data['name']}")
    print('='*50)

    if not create_library(client, library_id, library_data["name"]):
        return 0

    total_chunks = 0
    for doc_key, doc_data in library_data["documents"].items():
        doc_id = f"{library_key}_{doc_key}"
        if create_document(client, library_id, doc_id, doc_data["name"]):
            total_chunks += create_chunks(client, doc_id, doc_data["chunks"])

    return total_chunks


def main():
    parser = argparse.ArgumentParser(description="Seed StackAI with test data")
    parser.add_argument(
        "--library",
        choices=list(LIBRARIES.keys()) + ["all"],
        default="all",
        help="Library to seed (default: all)"
    )
    args = parser.parse_args()

    client = httpx.Client(base_url=BASE_URL, timeout=120.0)

    # Check server is running
    try:
        response = client.get("/libraries")
        if response.status_code != 200:
            print("Error: Could not connect to server. Is it running?")
            return
    except httpx.ConnectError:
        print("Error: Could not connect to server. Start it with: make start")
        return

    libraries_to_seed = list(LIBRARIES.keys()) if args.library == "all" else [args.library]

    total = 0
    for lib_key in libraries_to_seed:
        total += seed_library(client, lib_key)

    print(f"\n{'='*50}")
    print(f"Done! Created {total} chunks total.")
    print('='*50)

    # Show example searches
    print("\nExample searches to try:\n")
    examples = {
        "recipes": ("recipes_lib", "How do I make a creamy pasta sauce?"),
        "support": ("support_lib", "How do I reset my password?"),
        "products": ("products_lib", "How do I connect bluetooth headphones?"),
    }
    for lib_key in libraries_to_seed:
        lib_id, query = examples[lib_key]
        print(f"curl -X POST {BASE_URL}/libraries/{lib_id}/search \\")
        print(f"  -H 'Content-Type: application/json' \\")
        print(f"  -d '{{\"query\": \"{query}\", \"k\": 3}}'")
        print()


if __name__ == "__main__":
    main()
